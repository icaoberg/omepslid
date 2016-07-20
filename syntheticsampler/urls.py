import django.conf.urls

from syntheticsampler import views


urlpatterns = django.conf.urls.patterns('django.views.generic.simple',
    django.conf.urls.url(
      r'^image_ids\/*$', views.image_ids, name='syntheticsampler_image_ids'
    ),
    django.conf.urls.url(
      r'^images/(\d+)$', views.images, name='syntheticsampler_images'
    )
)
