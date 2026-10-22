"""Title-keyed structured experiences with progressive disclosure and staged CRUD."""

from copy import deepcopy

FIELDS = {"description", "observation", "experience"}


class ExperienceBank:
    def __init__(self, entries=None):
        self.entries = {}
        self.changes = []
        for title, entry in (entries or {}).items():
            self.upsert(title, entry)
        self.changes.clear()

    def directory(self, query=""):
        terms = query.casefold().split()
        return [
            {"title": title, "description": entry["description"]}
            for title, entry in sorted(self.entries.items())
            if all(
                term in (title + " " + entry["description"]).casefold()
                for term in terms
            )
        ]

    def read(self, title):
        if title not in self.entries:
            raise KeyError(f"Experience not found: {title}")
        return deepcopy(self.entries[title])

    def upsert(self, title, entry):
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError("Experience title must contain 1-200 characters")
        if not isinstance(entry, dict) or set(entry) != FIELDS:
            raise ValueError("Experience requires description, observation, experience")
        if any(
            not isinstance(v, str) or not v.strip() or len(v) > 16000
            for v in entry.values()
        ):
            raise ValueError(
                "Experience fields must be nonempty strings of at most 16000 characters"
            )
        before = deepcopy(self.entries.get(title))
        self.entries[title] = deepcopy(entry)
        self.changes.append(
            {
                "operation": "update" if before else "create",
                "title": title,
                "before": before,
                "after": deepcopy(entry),
            }
        )
        return {"title": title, "operation": self.changes[-1]["operation"]}

    def delete(self, title):
        before = self.read(title)
        del self.entries[title]
        self.changes.append(
            {"operation": "delete", "title": title, "before": before, "after": None}
        )
        return {"title": title, "operation": "delete"}

    def snapshot(self):
        return deepcopy(self.entries)
