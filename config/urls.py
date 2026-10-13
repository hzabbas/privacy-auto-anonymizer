from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('anonymizer.urls')),
]

# Static files are served by django.contrib.staticfiles while DEBUG is on; there
# is no need for an extra django.conf.urls.static() route, and nothing is served
# from here in production (run `collectstatic` instead).
