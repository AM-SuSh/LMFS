import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator

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

    pair_counts = count_pairs(pretoken_counts)
    pair_to_pretokens: dict[Pair, set[Pretoken]] = {}
    for pretoken in pretoken_counts:
        for pair in set(zip(pretoken, pretoken[1:])):
            pair_to_pretokens.setdefault(pair, set()).add(pretoken)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 同频时 bytes元组按字典序取大
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        changed_pretokens: Counter[Pretoken] = Counter()
        affected = list(pair_to_pretokens[best_pair])

        for pretoken in affected:
            frequency = pretoken_counts.pop(pretoken)
            old_pairs = Counter(zip(pretoken, pretoken[1:]))

            for pair, count in old_pairs.items():
                pair_counts[pair] -= count * frequency
                pair_to_pretokens[pair].discard(pretoken)

                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                if not pair_to_pretokens[pair]:
                    del pair_to_pretokens[pair]

            new_pretoken = merge_pair(pretoken, best_pair)
            changed_pretokens[new_pretoken] += frequency

        for pretoken, frequency in changed_pretokens.items():
            pretoken_counts[pretoken] += frequency
            new_pairs = Counter(zip(pretoken, pretoken[1:]))

            for pair, count in new_pairs.items():
                pair_counts[pair] += count * frequency
                pair_to_pretokens.setdefault(pair, set()).add(pretoken)

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

        self._refresh_indexes()

    def _refresh_indexes(self) -> None:
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens is not None:
            self.special_tokens = list(special_tokens)

        self.vocab, self.merges = train_bpe(input_path, vocab_size, self.special_tokens)
        self._refresh_indexes()

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

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        longest_special = max((len(token) for token in self.special_tokens), default=1)

        if self.special_tokens:
            ordered_tokens = sorted(self.special_tokens, key=len, reverse=True)
            specials = "|".join(regex.escape(token) for token in ordered_tokens)
            stream_pattern = f"(?:{specials})|(?:{PAT})"
        else:
            stream_pattern = PAT

        for chunk in iterable:
            buffer += chunk

            # 留下最后一个未结束的 pre-token，也留下可能构成特殊 token 的尾部。
            safe_limit = len(buffer) - longest_special + 1
            safe_end = 0
            for match in regex.finditer(stream_pattern, buffer):
                if match.start() <= safe_limit:
                    safe_end = match.start()
                else:
                    break

            if safe_end:
                yield from self.encode(buffer[:safe_end])
                buffer = buffer[safe_end:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, path_prefix: str | os.PathLike) -> None:
        data = {
            "vocab": [[token_id, token.hex()] for token_id, token in self.vocab.items()],
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(f"{path_prefix}.json", "w", encoding="utf-8", newline="") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def load(self, path_prefix: str | os.PathLike) -> None:
        with open(f"{path_prefix}.json", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = {int(token_id): bytes.fromhex(token) for token_id, token in data["vocab"]}
        self.merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in data["merges"]]
        self.special_tokens = list(data["special_tokens"])
        self._refresh_indexes()
