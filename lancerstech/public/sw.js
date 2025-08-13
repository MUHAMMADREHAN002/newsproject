self.addEventListener('install', (event) => {
  event.waitUntil(caches.open('static-v1').then((cache) => cache.add('/')));
});
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});