import os
from collections import Counter
import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pretoken = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    ordered_tokens = sorted(special_tokens, key=len, reverse=True)
    delimiter = "|".join(regex.escape(token) for token in ordered_tokens)
    return regex.split(delimiter, text)


def count_byte_pretokens(text: str, special_tokens: list[str]) -> Counter[Pretoken]:
    counts: Counter[Pretoken] = Counter()

    for segment in split_special_tokens(text, special_tokens):
        for match in regex.finditer(PAT, segment):
            utf8_bytes = match.group().encode("utf-8")
            byte_tuple = tuple(bytes([byte_value]) for byte_value in utf8_bytes)
            counts[byte_tuple] += 1

    return counts


def count_pairs(pretoken_counts: Counter[Pretoken]) -> Counter[Pair]:
    pair_counts: Counter[Pair] = Counter()

    for pretoken, frequency in pretoken_counts.items():
        for pair in zip(pretoken, pretoken[1:]):
            pair_counts[pair] += frequency

    return pair_counts


def merge_pair(pretoken: Pretoken, pair_to_merge: Pair) -> Pretoken:
    merged: list[bytes] = []
    index = 0

    while index < len(pretoken):
        if index + 1 < len(pretoken) and (pretoken[index], pretoken[index + 1]) == pair_to_merge:
            merged.append(pair_to_merge[0] + pair_to_merge[1])
            index += 2
        else:
            merged.append(pretoken[index])
            index += 1

    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    initial_vocab_size = len(special_tokens) + 256
    if vocab_size < initial_vocab_size:
        raise ValueError(f"vocab_size 至少应为 {initial_vocab_size}")

    vocab: dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for byte_value in range(256):
        vocab[len(vocab)] = bytes([byte_value])

    with open(input_path, encoding="utf-8", newline="") as file:
        text = file.read()

    pretoken_counts = count_byte_pretokens(text, special_tokens)
    merges: list[Pair] = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)
        if not pair_counts:
            break

        # 同频时 bytes元组按字典序取大
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        updated_counts: Counter[Pretoken] = Counter()
        for pretoken, frequency in pretoken_counts.items():
            updated_counts[merge_pair(pretoken, best_pair)] += frequency
        pretoken_counts = updated_counts

    return vocab, merges


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[Pair] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab or {})
        self.merges = list(merges or [])
        self.special_tokens = list(special_tokens or [])

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

    def _encode_pretoken(self, text: str) -> list[int]:
        pieces = [bytes([value]) for value in text.encode("utf-8")]

        while len(pieces) > 1:
            possible_merges = [
                (self.merge_rank[(left, right)], (left, right))
                for left, right in zip(pieces, pieces[1:])
                if (left, right) in self.merge_rank
            ]
            if not possible_merges:
                break

            _, pair = min(possible_merges)
            pieces = list(merge_pair(tuple(pieces), pair))

        return [self.token_to_id[piece] for piece in pieces]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        if self.special_tokens:
            ordered_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(regex.escape(token) for token in ordered_tokens)
            parts = regex.split(f"({special_pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
                continue

            for match in regex.finditer(PAT, part):
                ids.extend(self._encode_pretoken(match.group()))

        return ids

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")
