#!/usr/bin/env bash
base="https://cloud.overment.com/S01E01"
epoch=1730570331          # ten, który już mamy
for (( i=1; i<=60; i++ )); do
    try=$((epoch-i))
    url="${base}-${try}.png"
    echo -n "⏳ $url … "
    if curl -s --head "$url" | grep -q '200 OK'; then
        echo "FOUND"
        curl -s -o "old_$try.webp" "$url"
    else
        echo "404"
    fi
done