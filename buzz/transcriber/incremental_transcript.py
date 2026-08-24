from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class TranscriptUpdate:
    newly_committed_text: str
    provisional_text: str


@dataclass(frozen=True)
class _ComparisonUnit:
    value: str
    display_start: int
    display_end: int
    weight: int
    starts_source_span: bool
    ends_source_span: bool


@dataclass(frozen=True)
class _Anchor:
    previous_display_start: int
    previous_display_end: int
    current_display_start: int
    current_display_end: int


class IncrementalTranscript:
    """Stabilize text hypotheses produced from overlapping live audio.

    Committed text is append-only. Only the bounded provisional tail is
    reconciled with the next hypothesis. Full snapshots are constructed only
    when explicitly requested.
    """

    _MAX_ANCHOR_GAP_UNITS = 8
    _MIN_ANCHOR_SCORE = 4

    def __init__(
        self,
        *,
        max_revision_chars: int = 256,
        match_window_units: int = 128,
    ) -> None:
        if max_revision_chars <= 0:
            raise ValueError("max_revision_chars must be positive")
        if match_window_units <= 0:
            raise ValueError("match_window_units must be positive")

        self.max_revision_chars = max_revision_chars
        self.match_window_units = match_window_units
        self._committed_chunks: list[str] = []
        self._committed_tail = ""
        self._provisional = ""

    def update(self, hypothesis: str) -> TranscriptUpdate:
        if not hypothesis or not hypothesis.strip():
            return TranscriptUpdate("", self._provisional)

        current = hypothesis.strip()
        newly_committed = ""

        if self._provisional:
            anchor = self._find_anchor(self._provisional, current)
            if anchor is None:
                anchor = self._find_repeated_edge_anchor(
                    self._provisional,
                    current,
                )
            if anchor is None:
                newly_committed = self._provisional
                self._append_committed(newly_committed)
                self._provisional = self._safe_seam(newly_committed, current) + current
            else:
                newly_committed = self._provisional[:anchor.previous_display_end]
                self._append_committed(newly_committed)
                self._provisional = current[anchor.current_display_end:]
        elif self._committed_tail:
            # A hypothesis can end exactly at an anchor, leaving no visible
            # provisional text. Retain a bounded, immutable committed tail only
            # to suppress that same acoustic overlap on the next update.
            anchor = self._find_anchor(self._committed_tail, current)
            if anchor is None:
                anchor = self._find_repeated_edge_anchor(
                    self._committed_tail,
                    current,
                )
            if anchor is None:
                self._provisional = self._safe_seam(self._committed_tail, current) + current
            else:
                self._provisional = current[anchor.current_display_end:]
        else:
            self._provisional = current

        forced_commit = self._commit_provisional_overflow()
        newly_committed += forced_commit
        return TranscriptUpdate(newly_committed, self._provisional)

    def finalize(self) -> TranscriptUpdate:
        newly_committed = self._provisional
        self._append_committed(newly_committed)
        self._provisional = ""
        return TranscriptUpdate(newly_committed, "")

    def snapshot(self, *, include_provisional: bool = False) -> str:
        committed = "".join(self._committed_chunks)
        if include_provisional:
            return committed + self._provisional
        return committed

    def reset(self) -> None:
        self._committed_chunks.clear()
        self._committed_tail = ""
        self._provisional = ""

    def _append_committed(self, text: str) -> None:
        if not text:
            return

        self._committed_chunks.append(text)
        tail = self._committed_tail + text
        if len(tail) > self.max_revision_chars:
            tail = tail[-self.max_revision_chars:]
            while tail and _is_combining_mark(tail[0]):
                tail = tail[1:]
        self._committed_tail = tail

    def _commit_provisional_overflow(self) -> str:
        if len(self._provisional) <= self.max_revision_chars:
            return ""

        cut = len(self._provisional) - self.max_revision_chars
        while cut < len(self._provisional) and _is_combining_mark(
            self._provisional[cut]
        ):
            cut += 1

        newly_committed = self._provisional[:cut]
        self._provisional = self._provisional[cut:]
        self._append_committed(newly_committed)
        return newly_committed

    def _find_anchor(self, previous: str, current: str) -> _Anchor | None:
        previous_units = _comparison_units(previous)
        current_units = _comparison_units(
            current,
            max_units=self.match_window_units,
        )
        previous_units = previous_units[-self.match_window_units:]

        if not previous_units or not current_units:
            return None

        current_weight_prefix = [0]
        for unit in current_units:
            current_weight_prefix.append(current_weight_prefix[-1] + unit.weight)

        best_indices = None
        best_key = None

        # Walk every bounded comparison diagonal. Each equal-value run exposes
        # every possible end position, so repeated occurrences on either side
        # are considered instead of relying on one non-overlapping partition.
        for diagonal in range(-len(current_units) + 1, len(previous_units)):
            previous_index = max(diagonal, 0)
            current_index = max(-diagonal, 0)
            run_start: tuple[int, int] | None = None

            while (
                previous_index < len(previous_units)
                and current_index < len(current_units)
            ):
                previous_unit = previous_units[previous_index]
                current_unit = current_units[current_index]
                if previous_unit.value != current_unit.value:
                    run_start = None
                else:
                    if (
                        run_start is None
                        and previous_unit.starts_source_span
                        and current_unit.starts_source_span
                    ):
                        run_start = (previous_index, current_index)

                    if (
                        run_start is not None
                        and previous_unit.ends_source_span
                        and current_unit.ends_source_span
                    ):
                        previous_start, current_start = run_start
                        size = previous_index - previous_start + 1
                        leading_units = current_start
                        trailing_units = len(previous_units) - previous_index - 1
                        if (
                            leading_units <= self._MAX_ANCHOR_GAP_UNITS
                            and trailing_units <= self._MAX_ANCHOR_GAP_UNITS
                        ):
                            score = (
                                current_weight_prefix[current_index + 1]
                                - current_weight_prefix[current_start]
                            )
                            exact_edge = leading_units == 0 and trailing_units == 0
                            if score >= self._MIN_ANCHOR_SCORE and (
                                score > self._MIN_ANCHOR_SCORE or exact_edge
                            ):
                                key = (
                                    score,
                                    -leading_units,
                                    -trailing_units,
                                    size,
                                    -current_start,
                                    previous_start,
                                )
                                if best_key is None or key > best_key:
                                    best_key = key
                                    best_indices = (
                                        previous_start,
                                        previous_index,
                                        current_start,
                                        current_index,
                                    )

                previous_index += 1
                current_index += 1

        if best_indices is None:
            return None

        previous_start, previous_end, current_start, current_end = best_indices
        return _Anchor(
            previous_display_start=previous_units[previous_start].display_start,
            previous_display_end=previous_units[previous_end].display_end,
            current_display_start=current_units[current_start].display_start,
            current_display_end=current_units[current_end].display_end,
        )

    @staticmethod
    def _find_repeated_edge_anchor(previous: str, current: str) -> _Anchor | None:
        """Accept only an exact, doubled short token at the acoustic seam."""
        if not previous or not current:
            return None

        previous_last = previous[-1]
        if (
            _is_cjk_kana_hangul(previous_last)
            and len(current) >= 2
            and current[0] == previous_last
            and current[1] == previous_last
        ):
            return _Anchor(
                previous_display_start=len(previous) - 1,
                previous_display_end=len(previous),
                current_display_start=0,
                current_display_end=1,
            )

        previous_start = len(previous)
        while previous_start > 0 and _is_latin_alnum(previous[previous_start - 1]):
            previous_start -= 1
        previous_token = previous[previous_start:]
        if not previous_token:
            return None

        first_end = 0
        while first_end < len(current) and _is_latin_alnum(current[first_end]):
            first_end += 1
        first_token = current[:first_end]

        second_start = first_end
        while second_start < len(current) and current[second_start].isspace():
            second_start += 1
        if second_start == first_end:
            return None

        second_end = second_start
        while second_end < len(current) and _is_latin_alnum(current[second_end]):
            second_end += 1
        second_token = current[second_start:second_end]

        normalized_previous = _normalized_significant_text(previous_token)
        if len(normalized_previous) < 2 or not (
            normalized_previous == _normalized_significant_text(first_token)
            == _normalized_significant_text(second_token)
        ):
            return None

        return _Anchor(
            previous_display_start=previous_start,
            previous_display_end=len(previous),
            current_display_start=0,
            current_display_end=first_end,
        )

    @staticmethod
    def _safe_seam(left: str, right: str) -> str:
        if not left or not right:
            return ""

        left_char = left[-1]
        right_char = right[0]
        if left_char.isspace() or right_char.isspace():
            return ""
        if _is_punctuation_or_symbol(right_char):
            return ""
        if left_char in ",.!?;:" and not _is_cjk_kana_hangul(right_char):
            return " "
        if _is_punctuation_or_symbol(left_char):
            return ""
        if (
            left_char.isalnum()
            and right_char.isalnum()
            and not _is_cjk_kana_hangul(left_char)
            and not _is_cjk_kana_hangul(right_char)
        ):
            return " "
        return ""


def _comparison_units(
    text: str,
    *,
    max_units: int | None = None,
) -> list[_ComparisonUnit]:
    units: list[_ComparisonUnit] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and _is_combining_mark(text[end]):
            end += 1

        cluster = text[index:end]
        comparison_chars = [
            char
            for char in unicodedata.normalize("NFKC", cluster).casefold()
            if char.isalnum() or _is_cjk_kana_hangul(char)
        ]
        for comparison_index, char in enumerate(comparison_chars):
            units.append(
                _ComparisonUnit(
                    value=char,
                    display_start=index,
                    display_end=end,
                    weight=2 if _is_cjk_kana_hangul(char) else 1,
                    starts_source_span=comparison_index == 0,
                    ends_source_span=comparison_index == len(comparison_chars) - 1,
                )
            )
            if max_units is not None and len(units) >= max_units:
                return units
        index = end
    return units


def _normalized_significant_text(text: str) -> str:
    return "".join(unit.value for unit in _comparison_units(text))


def _is_latin_alnum(char: str) -> bool:
    return char.isalnum() and not _is_cjk_kana_hangul(char)


def _is_combining_mark(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def _is_punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char)[0] in {"P", "S"}


def _is_cjk_kana_hangul(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xFF66 <= codepoint <= 0xFF9D
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xD7B0 <= codepoint <= 0xD7FF
    )
