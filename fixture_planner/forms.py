from django import forms

class NameForm(forms.Form):
     Arsenal = forms.CharField(label='Arsenal', max_length=100)