# Generated by Django 2.2.4 on 2023-06-30 04:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0003_auto_20230630_0854'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedfile',
            name='file',
            field=models.FileField(upload_to='pdfs/'),
        ),
    ]
