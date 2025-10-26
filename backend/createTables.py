from app.core.database import engine, Base
from app.models.user import User

print("Creating database tables...")

Base.metadata.create_all(bind=engine)
print("Database tables created successfully! :)")
print("Database file: bot_detection.db")