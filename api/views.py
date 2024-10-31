from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action as restful_action


from .models import Patient, PatientImageData

from .serializers import PatientImageDataSerializer, PatientSerializer
from .yolo_model import ModelPredictor
from .genai import GenAI
import json


class PatientViewSet(ModelViewSet):

    def get_queryset(self):
        return Patient.objects.all()

    def get_serializer_class(self):
        return PatientSerializer

    def create(self, request, *args, **kwargs):

        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            serializer.save()

            response = {
                "created": True,
                "data": serializer.data,
            }
            return Response(response, status=status.HTTP_201_CREATED)

        else:

            response = {
                "created": False,
                "errors": serializer.errors,
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

    @restful_action(detail=True, methods=["GET"], url_path="get-predictions")
    def get_predictions(self, request, *args, **kwargs):
        patient_id = kwargs.get("pk")

        try:
            patient = Patient.objects.get(id=patient_id)

            model = ModelPredictor(patient.image.path)

            # make the prediction

            model.predict()
            # set the params
            model.set_params()

            # plot the image
            plotted_path = model.plot_image(patient_id)

            # generate heat map explanation
            model.generate_heatmap_explanation(patient_id)

            # get the params

            detection = model.get_params()[0]

            tumor_size = detection["tumor_size"]
            confidence = detection["confidence"]
            bounding_box = detection["bounding_box"]
            class_label = detection["class_label"]
            plotted_image = plotted_path

            # save the prediction

            patient_image_data = PatientImageData.objects.create(
                patient=patient,
                tumor_size=tumor_size,
                confidence=confidence,
                bounding_box=bounding_box,
                class_label=class_label,
            )

            patient_image_data.save()

            image_data = PatientImageDataSerializer(patient_image_data).data

            response = {
                "message": "Prediction generated successfully.",
                "data": image_data,
                "heatmap": f"media/results/heatmap/{patient_id}.jpg",
                "image": plotted_image,
                "tumor": True,
            }

            return Response(response, status=status.HTTP_200_OK)

        except Patient.DoesNotExist:
            response = {
                "message": "Patient not found",
                "found": False,
            }
            return Response(response, status=status.HTTP_404_NOT_FOUND)
        except IndexError or Exception:
            response = {
                "message": "No tumor present in the image",
                "tumor": False,
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

    @restful_action(detail=True, methods=["GET"], url_path="generate-treatment-plan")
    def generate_treatment_plan(self, request, *args, **kwargs):

        genai = GenAI()

        patient_id = kwargs.get("pk")

        try:

            patient = PatientImageData.objects.filter(patient_id=patient_id).first()

            data_dict = {
                "tumor_size": patient.tumor_size,
                "age": patient.patient.age,
                "sex": patient.patient.sex,
                "health_history": patient.patient.health_history,
                "prior_treatments": patient.patient.prior_treatments,
                "allergies": patient.patient.allergies,
                "existing_conditions": patient.patient.existing_conditions,
                "family_history": patient.patient.family_history,
            }

            # create a prompt
            prompt = genai.generate_prompt(data_dict)

            # create a thread

            thread_id = genai.create_thread()

            # create a message and run the thread

            response_text, message_status, message_id = genai.create_message(
                thread_id, str(prompt)
            )

            response = {
                "message": response_text,
                "message_status": message_status,
                "message_id": message_id,
            }

            return Response(response, status=status.HTTP_200_OK)

        except PatientImageData.DoesNotExist or Exception:
            response = {
                "message": "Patient Image Data not found",
                "found": False,
            }
            return Response(response, status=status.HTTP_404_NOT_FOUND)
