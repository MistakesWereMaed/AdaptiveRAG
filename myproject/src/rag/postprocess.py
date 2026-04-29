from __future__ import annotations

import re


def clean_answer(text: str) -> str:
	if text is None:
		return ""

	value = str(text).strip()
	if not value:
		return ""

	value = re.sub(r"^(answer|short answer)\s*:\s*", "", value, flags=re.IGNORECASE)
	value = value.splitlines()[0].strip()
	value = re.sub(r"\s+", " ", value)

	if value.endswith(".") and len(value.split()) <= 8:
		value = value[:-1].rstrip()

	return value