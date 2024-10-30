from django.urls import path, include

from rest_framework.routers import DefaultRouter
from .views import PatientViewSet

router = DefaultRouter()

# register the patient viewset
router.register("patients", PatientViewSet, basename="patient")

urlpatterns = [
    path("", include(router.urls)),
]
