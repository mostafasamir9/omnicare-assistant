# syntax=docker/dockerfile:1.7

FROM nginx:1.27-alpine

# Custom nginx config: SPA-friendly, gzip, security headers
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Static assets
COPY frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]