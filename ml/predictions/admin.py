from django.contrib import admin
from django.utils.html import format_html
from .models import UploadsImage, MisclassifiedImage

@admin.register(UploadsImage)
class UploadsImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'image_tag',)  # Show thumbnail instead of just file name
    search_fields = ('id',)

    def image_tag(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" />', obj.image.url)
        return "-"
    image_tag.short_description = 'Image'  # Column name

@admin.register(MisclassifiedImage)
class MisclassifiedImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'image_tag',)
    search_fields = ('id',)

    def image_tag(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" />', obj.image.url)
        return "-"
    image_tag.short_description = 'Image'
