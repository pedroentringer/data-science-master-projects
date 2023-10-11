import threading
from json import dumps
from time import sleep
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'))

def producer_one(topic):
    for e in range(1000):
        data = {'producer-one-number': e}
        print(data)
        producer.send(topic, value=data)
        sleep(5)

def producer_two(topic):
    for e in range(1000):
        data = {'producer-two-number': e}
        print(data)
        producer.send(topic, value=data)
        sleep(5)

topic = "random_numbers"

t1 = threading.Thread(target=producer_one, args=(topic,))
t2 = threading.Thread(target=producer_two, args=(topic,))
t1.start()
t2.start()

