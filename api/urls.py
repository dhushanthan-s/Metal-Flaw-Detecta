from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path(r'^users/$', views.UsersView.as_view(), name='user'),
    path(r'^users/([0-9])$/dashboard', views.dashboard, name='user_dashboard'),
    path(r'predict/', views.predict, name='predict'),
    path(r'models/$', views.model_info, name='model_info'),
    path(r'models/', views.models, name="models"),
    path(r'training/', views.training, name='training'),
] + static(settings.DATASET_URL, document_root=settings.DATASET_ROOT)
