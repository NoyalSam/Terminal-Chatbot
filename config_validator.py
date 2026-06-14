from config import settings

def validate_config():
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY missing in environment variables")

    print("Configuration validated successfully.")