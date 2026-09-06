from django.shortcuts import render

def index(request):
    """
    Renders the main single-page interface for Privacy Auto-Anonymizer.
    """
    return render(request, 'anonymizer/index.html')
