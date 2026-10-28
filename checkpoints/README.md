# checkpoints/

Обычно эта папка пустая: веса `scaled_offline.pth` скачиваются автоматически при сборке
Docker-образа (stage `weights` в `Dockerfile`): сначала с Hugging Face, а если его CDN
(`*.hf.co`) недоступен — с зеркала ModelScope.

Если при сборке видно `[weights] download failed` (недоступны оба источника), скачайте
файл вручную (браузером, через VPN, с другой машины) по любой из ссылок:

- https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
- https://www.modelscope.cn/models/facebook/cotracker3/resolve/master/scaled_offline.pth

и положите его сюда как `checkpoints/scaled_offline.pth`. Затем пересоберите образ
(`bash run.sh --rebuild`). Файл будет проверен:

- размер: `101890938` байт
- sha256: `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`

Файл в git не попадает (см. `.gitignore`).
