import json
import heapq
import regex as re
from utils.common import gpt2_bytes_to_unicode


class Node:
    """双向链表的节点"""
    def __init__(self, token_id, bytes_seq):
        self.id = token_id       # token ID
        self.bytes = bytes_seq   # 原始 bytes
        self.prev = None
        self.next = None
        self.alive = True
        self.position = 0        # 节点在链表中的位置

    def kill(self):
        self.alive = False


class Pair:
    """堆中存储的二元组"""
    def __init__(self, left, right, rank):
        self.left = left
        self.right = right
        self.rank = rank

    def __lt__(self, other):
        # 首先按 rank 排序
        if self.rank != other.rank:
            return self.rank < other.rank
        # rank 相同时，按左节点的位置排序（保证从左到右）
        return self.left.position < other.left.position

    def valid(self):
        return (
            self.left.alive and
            self.right.alive and
            self.left.next is self.right
        )


class FastTokenizer:
    def __init__(self, vocab_path, merges_path, special_tokens=None):
        """
        vocab_path: GPT-2 格式的 vocab.json 文件路径
        merges_path: GPT-2 格式的 merges.txt 文件路径
        special_tokens: 特殊 token 列表，如 ["<|endoftext|>"]
        """
        # 使用正确的 GPT-2 bytes_to_unicode 映射
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        
        # 存储 special tokens
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_set = set(self.special_tokens)
        self.special_token_bytes = {token: token.encode('utf-8') for token in self.special_tokens}
        
        # GPT-2 预分词正则表达式
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 加载 vocab - GPT-2 格式 {unicode_string: token_id}
        # 转换为 {bytes: token_id} 格式
        with open(vocab_path, 'r', encoding='utf-8') as f:
            gpt2_vocab = json.load(f)
        
        self.bytes_to_id = {}
        self.id_to_bytes = {}
        
        for token_str, token_id in gpt2_vocab.items():
            # 将 GPT-2 unicode string 转换回 bytes
            token_bytes = bytes([byte_decoder[c] for c in token_str])
            self.bytes_to_id[token_bytes] = token_id
            self.id_to_bytes[token_id] = token_bytes
        
        # 加载 merges - 使用 byte_decoder 转换（和 vocab 保持一致）
        self.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) == 2:
                    a_str, b_str = line.split()
                    # 使用 byte_decoder 转换每个字符
                    a_bytes = bytes([byte_decoder[c] for c in a_str])
                    b_bytes = bytes([byte_decoder[c] for c in b_str])
                    self.merges.append((a_bytes, b_bytes))
        
        # 构建 pair_rank - 使用 bytes 作为 key
        self.pair_rank = {
            (a, b): i for i, (a, b) in enumerate(self.merges)
        }

    def bytes_to_nodes(self, text_bytes):
        """将原始 bytes 转为初始节点链表"""
        nodes = []
        for i, b in enumerate(text_bytes):
            single_byte = bytes([b])
            token_id = self.bytes_to_id.get(single_byte)
            
            if token_id is None:
                raise ValueError(f"Byte {b} not in vocab")
            
            node = Node(token_id, single_byte)
            node.position = i  # 设置初始位置
            nodes.append(node)

        # 链接节点
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i+1]
            nodes[i+1].prev = nodes[i]

        return nodes[0] if nodes else None

    def merge_pair(self, left, right):
        """合并 pair: 将 left 和 right 合并为 new_node"""
        new_bytes = left.bytes + right.bytes
        new_id = self.bytes_to_id.get(new_bytes)

        if new_id is None:
            raise ValueError(f"Missing vocab entry for merged token {new_bytes}")

        # 创建新节点
        new_node = Node(new_id, new_bytes)
        # 继承左节点的位置
        new_node.position = left.position

        # 链表更新
        L = left.prev
        R = right.next

        new_node.prev = L
        new_node.next = R

        if L:
            L.next = new_node
        if R:
            R.prev = new_node

        # 原节点失效
        left.kill()
        right.kill()

        return new_node

    def add_pair(self, pq, left, right):
        """将邻居 pair 放入堆"""
        if left and right and left.alive and right.alive:
            # 使用 bytes 作为 pair key
            pair = (left.bytes, right.bytes)
            
            if pair in self.pair_rank:
                rank = self.pair_rank[pair]
                heapq.heappush(pq, Pair(left, right, rank))

    def encode_pretoken(self, text_bytes):
        """对单个预分词的 bytes 进行 BPE 编码"""
        # 优化：先检查整个 token 是否已经在 vocab 中
        if text_bytes in self.bytes_to_id:
            return [self.bytes_to_id[text_bytes]]
        
        head = self.bytes_to_nodes(text_bytes)
        if head is None:
            return []

        # 初始化堆：扫描所有邻接 pair
        pq = []
        node = head
        while node and node.next:
            self.add_pair(pq, node, node.next)
            node = node.next

        # 主循环：不断合并 rank 最小的 pair
        while pq:
            p = heapq.heappop(pq)

            if not p.valid():
                continue

            # 合并
            new_node = self.merge_pair(p.left, p.right)

            # 更新 neighbors
            self.add_pair(pq, new_node.prev, new_node)
            self.add_pair(pq, new_node, new_node.next)

            # 如果此节点成为新头部，更新 head
            if new_node.prev is None:
                head = new_node

        # 输出所有存活节点的 token id
        tokens = []
        node = head
        while node:
            tokens.append(node.id)
            node = node.next

        return tokens

    def _split_by_special_tokens(self, text):
        """
        按 special tokens 分割文本
        返回：[(text_or_token, is_special), ...]
        """
        if not self.special_tokens:
            return [(text, False)]
        
        # 转义 special tokens 用于正则表达式
        escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
        pattern = "|".join(escaped_tokens)
        
        # 分割文本
        parts = re.split(f"({pattern})", text)
        
        result = []
        for part in parts:
            if part:  # 忽略空字符串
                is_special = part in self.special_tokens_set
                result.append((part, is_special))
        
        return result

    def encode(self, text):
        """编码：Text → Token IDs (带预分词和 special tokens 支持)"""
        # 按 special tokens 分割
        parts = self._split_by_special_tokens(text)
        
        all_token_ids = []
        for part, is_special in parts:
            if is_special:
                # 处理 special token
                special_bytes = self.special_token_bytes[part]
                token_id = self.bytes_to_id.get(special_bytes)
                if token_id is not None:
                    all_token_ids.append(token_id)
                else:
                    # Special token 不在 vocab 中，按普通文本处理
                    pre_tokens = self.pat.findall(part)
                    for pre_token in pre_tokens:
                        pre_token_bytes = pre_token.encode('utf-8')
                        token_ids = self.encode_pretoken(pre_token_bytes)
                        all_token_ids.extend(token_ids)
            else:
                # 普通文本：预分词
                pre_tokens = self.pat.findall(part)
                for pre_token in pre_tokens:
                    pre_token_bytes = pre_token.encode('utf-8')
                    token_ids = self.encode_pretoken(pre_token_bytes)
                    all_token_ids.extend(token_ids)
        
        return all_token_ids

    def decode(self, token_ids):
        """解码 Token IDs → 文本"""
        out = bytearray()
        for tid in token_ids:
            token_bytes = self.id_to_bytes[tid]
            out.extend(token_bytes)
        return out.decode("utf-8", errors="replace")
    
    def decode_bytes(self, token_ids):
        """解码 Token IDs → bytes (无损)"""
        out = bytearray()
        for tid in token_ids:
            token_bytes = self.id_to_bytes[tid]
            out.extend(token_bytes)
        return bytes(out)
    
    def encode_bytes(self, data_bytes):
        """
        从 bytes 直接编码 (无损，不经过预分词)
        用于保证 round-trip 一致性
        """
        # 直接对整个 byte 序列进行 BPE，不经过文本预分词
        return self.encode_pretoken(data_bytes)
    
    def decode_bytes(self, token_ids):
        """解码 Token IDs → bytes (无损)"""
        out = bytearray()
        for tid in token_ids:
            token_bytes = self.id_to_bytes[tid]
            out.extend(token_bytes)
        return bytes(out)
    
    def encode_bytes(self, data_bytes):
        """
        从 bytes 直接编码 (无损，不经过文本解码)
        这样可以避免 round-trip 不一致问题
        """
        # 将 bytes 转换为 token IDs，不经过预分词
        # 这是原始的 byte-level BPE
        return self.encode_pretoken(data_bytes)