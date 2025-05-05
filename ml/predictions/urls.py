from django.urls import path
from .views import PredictImageView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("predict/", PredictImageView.as_view(), name="predict-image"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
