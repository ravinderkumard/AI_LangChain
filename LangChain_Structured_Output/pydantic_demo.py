from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    company: str = "Default Company"
    email: EmailStr
    cgpa: float = Field(default=6, ge=0.0, le=10.0, description="CGPA must be between 0.0 and 10.0")


new_student = {'name': 'John Doe', 'age': 20, 'email': 'ab@gmail.com', 'cgpa': 8}

student = Student(**new_student)

print(student)

student_dict = dict(student)
print(student_dict)

student_json = student.model_dump_json()

print(student_json)
