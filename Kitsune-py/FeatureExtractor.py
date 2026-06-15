# Check if cython code has been compiled
import os
import subprocess
import sys
import platform

# Import dependencies
import csv
import numpy as np
import netStat as ns
print("Importing Scapy Library")
from scapy.all import rdpcap, PcapReader, IP, IPv6, TCP, UDP, ARP, ICMP
import os.path


# Extracts Kitsune features from given pcap/tsv/csv file one packet at a time using "get_next_vector()"
class FE:
    def __init__(self, file_path, limit=np.inf):
        self.path = file_path
        self.limit = limit
        self.parse_type = None  # 'tsv', 'csv', or 'scapy'
        self.curPacketIndx = 0
        self.tsvin = None
        self.tsvinf = None
        self.scapyin = None

        ### Prep pcap/csv/tsv ##
        self.__prep__()

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return 'C:\\Program Files\\Wireshark\\tshark.exe'
        else:
            system_path = os.environ.get('PATH', '')
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def __prep__(self):
        # verify file exists
        if not os.path.isfile(self.path):
            raise Exception("File: {} does not exist".format(self.path))

        ftype = self.path.split('.')[-1].lower()
        self._tshark = self._get_tshark_path()

        if ftype == 'tsv' or ftype == 'csv':
            self.parse_type = ftype
            self.delim = '\t' if ftype == 'tsv' else ','

            # handle very large fields
            maxInt = sys.maxsize
            decrement = True
            while decrement:
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt / 10)
                    decrement = True

            # Open reader and skip header — do NOT pre-count lines so we can stream large files.
            self.tsvinf = open(self.path, 'rt', encoding='utf8', errors='ignore')
            self.tsvin = csv.reader(self.tsvinf, delimiter=self.delim)
            try:
                next(self.tsvin)
            except StopIteration:
                pass

            # Respect user-provided limit if finite, otherwise leave as np.inf to read until EOF
            if self.limit != np.inf:
                self.limit = int(self.limit)

        elif ftype == 'pcap' or ftype == 'pcapng':
            # prefer tshark conversion when available (faster)
            if os.path.isfile(self._tshark):
                self.pcap2tsv_with_tshark()
                self.path += '.tsv'
                self.parse_type = 'tsv'
                self.delim = '\t'
                # reuse tsv path handling
                with open(self.path, 'r', encoding='utf8', errors='ignore') as fh:
                    num_lines = sum(1 for _ in fh)
                if self.limit == np.inf:
                    self.limit = max(0, num_lines - 1)
                else:
                    self.limit = min(self.limit, max(0, num_lines - 1))
                self.tsvinf = open(self.path, 'rt', encoding='utf8', errors='ignore')
                self.tsvin = csv.reader(self.tsvinf, delimiter=self.delim)
                try:
                    next(self.tsvin)
                except StopIteration:
                    pass
            else:
                print("tshark not found. Trying scapy...")
                self.parse_type = 'scapy'
                # scapy path: stream or load
                if self.limit == np.inf:
                    self.scapyin = rdpcap(self.path)
                    self.limit = len(self.scapyin)
                    print("Loaded {} Packets.".format(len(self.scapyin)))
                else:
                    # open a PcapReader for true streaming — do not cache packets in memory
                    try:
                        self.scapy_reader = PcapReader(self.path)
                        self.scapyin = None
                        self._scapy_streaming = True
                        # ensure integer limit
                        self.limit = int(self.limit)
                        print("Opened pcap for streaming (limit={}).".format(self.limit))
                    except Exception:
                        # fallback to buffering if PcapReader fails
                        self.scapyin = rdpcap(self.path)
                        self.scapyin = self.scapyin[:int(self.limit)]
                        self.limit = len(self.scapyin)
                        print("Loaded {} Packets (fallback).".format(len(self.scapyin)))
        else:
            raise Exception("Unsupported file type: {}".format(ftype))

    def get_next_vector(self):
        # finished
        if self.curPacketIndx >= self.limit:
            if self.parse_type in ('tsv', 'csv') and self.tsvinf is not None:
                try:
                    self.tsvinf.close()
                except Exception:
                    pass
            return []

        if self.parse_type in ('tsv', 'csv'):
            try:
                row = next(self.tsvin)
            except Exception:
                return []

            IPtype = np.nan
            timestamp = row[0]
            framelen = row[1]
            srcIP = ''
            dstIP = ''
            if row[4] != '':  # IPv4
                srcIP = row[4]
                dstIP = row[5]
                IPtype = 0
            elif row[17] != '':  # ipv6
                srcIP = row[17]
                dstIP = row[18]
                IPtype = 1

            srcproto = row[6] + row[8]
            dstproto = row[7] + row[9]
            srcMAC = row[2]
            dstMAC = row[3]

            if srcproto == '':
                if row[12] != '':  # is ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = row[14]
                    dstIP = row[16]
                    IPtype = 0
                elif row[10] != '':  # is ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':
                    srcIP = row[2]
                    dstIP = row[3]

        elif self.parse_type == 'scapy':
            # If a PcapReader was opened for streaming, read one pkt from it without caching
            if getattr(self, 'scapy_reader', None) is not None:
                try:
                    packet = next(self.scapy_reader)
                except StopIteration:
                    try:
                        self.scapy_reader.close()
                    except Exception:
                        pass
                    return []
            else:
                # buffered mode: packets were loaded into memory
                packet = self.scapyin[self.curPacketIndx]

            IPtype = np.nan
            timestamp = packet.time
            framelen = len(packet)

            if packet.haslayer(IP):
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                IPtype = 0
            elif packet.haslayer(IPv6):
                srcIP = packet[IPv6].src
                dstIP = packet[IPv6].dst
                IPtype = 1
            else:
                srcIP = ''
                dstIP = ''

            if packet.haslayer(TCP):
                srcproto = str(packet[TCP].sport)
                dstproto = str(packet[TCP].dport)
            elif packet.haslayer(UDP):
                srcproto = str(packet[UDP].sport)
                dstproto = str(packet[UDP].dport)
            else:
                srcproto = ''
                dstproto = ''

            srcMAC = getattr(packet, 'src', '')
            dstMAC = getattr(packet, 'dst', '')
            if srcproto == '':
                if packet.haslayer(ARP):
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = packet[ARP].psrc
                    dstIP = packet[ARP].pdst
                    IPtype = 0
                elif packet.haslayer(ICMP):
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':
                    srcIP = getattr(packet, 'src', '')
                    dstIP = getattr(packet, 'dst', '')
        else:
            return []

        self.curPacketIndx += 1

        try:
            return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                             int(framelen), float(timestamp))
        except Exception as e:
            print(e)
            return []

    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd = '"' + self._tshark + '" -r ' + self.path + ' -T fields ' + fields + ' -E header=y -E occurrence=f > ' + self.path + ".tsv"
        subprocess.call(cmd, shell=True)
        print("tshark parsing complete. File saved as: " + self.path + ".tsv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
