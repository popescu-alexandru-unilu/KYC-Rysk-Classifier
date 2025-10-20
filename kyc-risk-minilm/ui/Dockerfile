FROM nginx:alpine
ARG IS_LOCAL=false
WORKDIR /usr/share/nginx/html
COPY . .
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN echo "window.IS_LOCAL = ${IS_LOCAL};" > /usr/share/nginx/html/env.js
EXPOSE 80
