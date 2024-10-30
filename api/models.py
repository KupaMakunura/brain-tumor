import uuid
from django.db import models


# Create your models here.
class Patient(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    age = models.CharField(max_length=100)
    sex = models.CharField(max_length=100)
    health_history = models.CharField(max_length=100, null=True)
    prior_treatments = models.CharField(max_length=100, null=True)
    allergies = models.CharField(max_length=100, null=True)
    existing_conditions = models.CharField(max_length=100, null=True)
    family_history = models.CharField(max_length=100, null=True)
    image = models.ImageField(upload_to="images/", null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
