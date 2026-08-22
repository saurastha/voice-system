import json
from agent.database import SessionLocal, Booking, RestaurantConfig


def get_restaurant_info() -> str:
    """Fetch restaurant operating hours, location, contact details, and daily specials."""
    db = SessionLocal()
    try:
        config = db.query(RestaurantConfig).first()
        if config and config.info_data:
            return config.info_data
        return json.dumps({"info": "Restaurant details currently unavailable."})
    finally:
        db.close()


class BookingSession:
    """Tracks booking slots per user session to prevent duplicate questions."""

    def __init__(self):
        # We keep the draft in memory! It's lightning fast and tied to this specific caller's WebSocket connection.
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
        """Call this tool WHENEVER the guest provides ANY booking details."""

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

        db = SessionLocal()
        try:
            new_booking = Booking(
                date=self.draft["date"],
                time=self.draft["time"],
                party_size=self.draft["party_size"],
                guest_name=self.draft["guest_name"],
                phone_number=self.draft["phone_number"],
                high_chair=self.draft["high_chair"],
            )

            db.add(new_booking)
            db.commit()

            booking_id = new_booking.id

            return (
                f"SUCCESS: Table booked for {self.draft['guest_name']} "
                f"on {self.draft['date']} at {self.draft['time']} for {self.draft['party_size']} guests. "
                f"Confirmation number is #{booking_id}."
            )
        except Exception as e:
            db.rollback()
            return f"ERROR: Could not save the booking due to a database error: {e}"
        finally:
            db.close()
