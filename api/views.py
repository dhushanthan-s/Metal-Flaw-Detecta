from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from .models import Users
from .serializers import UserSerializer

# Create your views here.
class UsersView(generics.CreateAPIView):
    queryset = Users.objects.all()
    serializer_class = UserSerializer

