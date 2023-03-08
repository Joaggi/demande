from datetime import datetime
def get_current_day():
    today = datetime.now()
    # dd/mm/YY
    current_date = today.strftime("%Y%m%d%H%M%S")
    return current_date
