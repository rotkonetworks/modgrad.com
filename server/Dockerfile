FROM node:20-alpine
WORKDIR /app
COPY package.json contact.mjs ./
ENV NODE_ENV=production
EXPOSE 3001
USER node
CMD ["node", "contact.mjs"]
