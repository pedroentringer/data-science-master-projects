from json import loads
from kafka import KafkaConsumer

consumer = KafkaConsumer('random_numbers',bootstrap_servers=['localhost:9092'], auto_offset_reset='earliest', enable_auto_commit=True, group_id='mygroup', value_deserializer = lambda x: loads (x.decode( 'utf-8')))

for message in consumer:
    message = message.value
    print(message)