# analysis/urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('analyze/<int:file_id>/', views.analyze_file, name='analyze_file'),
    path('download/<int:file_id>/<str:format>/', views.download_highlighted_images, name='download_highlighted_images'),
    path('download_word_file/<int:file_id>/', views.download_word_file, name='download_word_file'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
