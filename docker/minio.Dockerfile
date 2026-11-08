# MinIO Community Edition and mc are built from pinned upstream source rather
# than relying on a stale, prebuilt :latest image. Both upstream repositories
# are archived; see docs/deployment.md before using them in production.
FROM golang:1.24-bookworm AS build
ARG MINIO_REV=7aac2a2c5b7c882e68c1ce017d8256be2feea27f
ARG MC_REV=77f82e18b5401a65958f1619df6ebb994634bd88
RUN mkdir -p /out \
    && git clone https://github.com/minio/minio.git /src/minio \
    && git -C /src/minio checkout --detach "${MINIO_REV}" \
    && cd /src/minio \
    && CGO_ENABLED=0 go build -tags kqueue -trimpath -o /out/minio .
RUN git clone https://github.com/minio/mc.git /src/mc \
    && git -C /src/mc checkout --detach "${MC_REV}" \
    && cd /src/mc \
    && CGO_ENABLED=0 go build -trimpath -o /out/mc .

FROM alpine:3.22
RUN apk add --no-cache ca-certificates \
    && addgroup -g 10001 sorani \
    && adduser -D -u 10001 -G sorani sorani \
    && mkdir -p /data \
    && chown 10001:10001 /data
COPY --from=build /out/minio /usr/local/bin/minio
COPY --from=build /out/mc /usr/local/bin/mc
USER 10001:10001
ENV HOME=/home/sorani
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
