import json
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///restaurant.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, index=True)
    time = Column(String)
    party_size = Column(Integer)
    guest_name = Column(String)
    phone_number = Column(String)
    high_chair = Column(String)


class RestaurantConfig(Base):
    """Stores general info about the restaurant (hours, location, etc)"""

    __tablename__ = "restaurant_config"

    id = Column(Integer, primary_key=True, index=True)
    info_data = Column(Text)


Base.metadata.create_all(bind=engine)


def seed_restaurant_info():
    """Seeds the DB with initial restaurant data if it's empty."""
    db = SessionLocal()
    if not db.query(RestaurantConfig).first():
        default_info = {
            "name": "The Winston",
            "location": "123 Sydney Harbour, Australia",
            "operating_hours": "Monday - Sunday: 5 PM to 11 PM",
            "daily_special": "Wood-fired Truffle Pizza",
            "contact_email": "hello@thewinston.com",
        }
        config = RestaurantConfig(info_data=json.dumps(default_info))
        db.add(config)
        db.commit()
    db.close()


seed_restaurant_info()
