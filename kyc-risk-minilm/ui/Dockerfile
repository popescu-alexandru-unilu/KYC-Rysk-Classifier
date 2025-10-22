FROM nginx:alpine
ARG IS_LOCAL=false
WORKDIR /usr/share/nginx/html
COPY . .
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
