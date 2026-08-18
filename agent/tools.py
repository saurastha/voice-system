import json
import os

RESTAURANT_DB_PATH = "data.json"
BOOKING_DB_PATH = "bookings.json"


def get_restaurant_info() -> str:
    """Fetch restaurant operating hours, location, contact details, and daily specials."""
    try:
        with open(RESTAURANT_DB_PATH, "r") as f:
            data = json.load(f)
        return json.dumps(data)
    except FileNotFoundError:
        return json.dumps({"info": "Restaurant data file not found."})


class BookingSession:
    """Tracks booking slots per user session to prevent duplicate questions."""

    def __init__(self):
        self.draft = {
            "date": None,
            "time": None,
            "party_size": None,
            "guest_name": None,
            "phone_number": None,
            "high_chair": None,
        }

    def record_booking_details(
        self,
        date: str = None,
        time: str = None,
        party_size: int = None,
        guest_name: str = None,
        phone_number: str = None,
        high_chair: str = None,
    ) -> str:
        """Call this tool WHENEVER the guest provides ANY booking details.

        Args:
            date: Date of booking (e.g., 'Today', 'Tomorrow', 'Friday', '2026-08-20')
            time: Time of booking (e.g., '7 PM', '19:30')
            party_size: Number of people/guests (e.g., 2, 4)
            guest_name: Full name of the guest
            phone_number: Contact phone number
            high_chair: High chair requirement ('yes', 'no', or 'not needed')
        """
        if date:
            self.draft["date"] = date
        if time:
            self.draft["time"] = time
        if party_size:
            self.draft["party_size"] = party_size
        if guest_name:
            self.draft["guest_name"] = guest_name
        if phone_number:
            self.draft["phone_number"] = phone_number
        if high_chair is not None:
            self.draft["high_chair"] = high_chair

        # Find missing items
        missing = [k for k, v in self.draft.items() if v is None]
        collected = {k: v for k, v in self.draft.items() if v is not None}

        if not missing:
            return (
                f"ALL DETAILS COLLECTED: {collected}. "
                f"Please briefly confirm these key details with the guest before calling book_table."
            )

        return (
            f"SAVED DETAILS: {collected}. "
            f"STILL MISSING: {missing}. "
            f"Ask ONLY for the missing details."
        )

    def book_table(self) -> str:
        """Finalize and save the reservation to the database. Call this ONLY after all details are recorded AND the guest confirms."""
        missing = [k for k, v in self.draft.items() if v is None]
        if missing:
            return f"CANNOT BOOK YET. Missing required details: {missing}."

        try:
            if os.path.exists(BOOKING_DB_PATH):
                with open(BOOKING_DB_PATH, "r") as f:
                    data = json.load(f)
            else:
                data = []
        except Exception:
            data = []

        data.append(self.draft.copy())

        with open(BOOKING_DB_PATH, "w") as f:
            json.dump(data, f, indent=4)

        return (
            f"SUCCESS: Table booked for {self.draft['guest_name']} "
            f"on {self.draft['date']} at {self.draft['time']} for {self.draft['party_size']} guests."
        )
