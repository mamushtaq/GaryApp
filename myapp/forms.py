from django import forms
from .models import UploadedFile

class ChatForm(forms.Form):
    body = forms.CharField(widget=forms.Textarea, required=False)
    msg = forms.CharField(widget=forms.TextInput(), required=False)

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ('file',)