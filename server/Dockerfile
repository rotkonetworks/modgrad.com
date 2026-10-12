FROM node:22-alpine
WORKDIR /app
# contact.mjs is what runs; ledger.mjs + proxy.mjs ship too, for `node proxy.mjs`
COPY package.json contact.mjs ledger.mjs proxy.mjs ./
ENV NODE_ENV=production
EXPOSE 3001
USER node
CMD ["node", "contact.mjs"]
