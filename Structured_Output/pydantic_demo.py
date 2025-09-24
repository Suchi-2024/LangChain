from pydantic import BaseModel, EmailStr, Field
from typing import Optional
 
class Student(BaseModel):
    name: str ="Anonymous"
    age : int
    gender: Optional[str]= "Don't want to specify"
    cgpa: float= Field(gt=0.0, lt=10.0, description="CGPA should be transformed into 10 scale", default=0.3)
    email: EmailStr

new_student={'name': 'Suchi', 'age': '22', 'gender': 'Female', 'email': 'abd@gmail.com', 'cgpa': 4.2}

student=Student(**new_student)

print(student)

print(student.model_dump())