from twilio.rest import Client
account_sid = ''
auth_token = ''
twilio_phone_number = ''
recipient_phone_number = ''

client = Client(account_sid, auth_token)

def alert():
    message = client.messages.create(
        body="Violence detected",
        from_=twilio_phone_number,
        to=recipient_phone_number
    )
    return "Message sent successfully!"
