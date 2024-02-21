from twilio.rest import Client
account_sid = 'AC7e613443a79f459e69da555245d4c46a'
auth_token = '1ada33a7725e1937fc3e70e3942d7662'
twilio_phone_number = '+16592702574'
recipient_phone_number = '+919495076875'

client = Client(account_sid, auth_token)

def alert():
    message = client.messages.create(
        body="Violence detected",
        from_=twilio_phone_number,
        to=recipient_phone_number
    )
    return "Message sent successfully!"