from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action as restful_action

from .models import Patient, PatientImageData

from .serializers import PatientImageDataSerializer, PatientSerializer
from .yolo_model import ModelPredictor


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
            plotted_path = model.plot_image()

            # generate heat map explanation
            model.generate_heatmap_explanation()

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
                "plotted_image": plotted_image,
            }

            return Response(response, status=status.HTTP_200_OK)

        except Patient.DoesNotExist:
            response = {
                "message": "Patient not found",
                "found": False,
            }
            return Response(response, status=status.HTTP_404_NOT_FOUND)

    @restful_action(detail=True, methods=["GET"], url_path="generate-treatment-plan")
    def generate_treatment_plan(self, request, *args, **kwargs):

        response = {
            "message": "Treatment plan generated successfully.",
        }

        return Response(response, status=status.HTTP_200_OK)
