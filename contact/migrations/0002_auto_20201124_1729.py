# Generated by Django 3.1.3 on 2020-11-24 16:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contact', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='webpageownerinfo',
            name='owner_description',
            field=models.CharField(default=1, help_text='Enter small description of yourself', max_length=400),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='webpageownerinfo',
            name='owner_twitter',
            field=models.CharField(default=1, help_text='Enter twitter url', max_length=40),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='webpageownerinfo',
            name='owner_name',
            field=models.CharField(help_text='Enter owner of webpage', max_length=30),
        ),
    ]
