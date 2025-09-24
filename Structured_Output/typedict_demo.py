from typing import TypedDict

class Person(TypedDict):
    name: str
    age : int
    city: str

new_person : Person = {'name':'Suchi', 'age': 22, 'city': 'Kolkata'}

print(new_person)