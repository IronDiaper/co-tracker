# TESTING.md — как проверить работу

## Кратко

```bash
git clone https://github.com/ironDiaper/co-tracker.git
cd co-tracker
bash run.sh            # Windows PowerShell: .\run.ps1
```

Ожидаемый конец вывода: таблица `CoTracker3: results` с 13 строками `PASSED`,
`13 passed`, список файлов в `./results/tests` и `TESTS DONE!`. Видео и картинки
открываются прямо из `./results/tests/` на хосте.

## Требования

| Ресурс | Минимум | Замечания |
|---|---|---|
| CPU | x86_64, 4 ядра | работает и на 2 ядрах, но тесты идут вдвое дольше |
| RAM | 8 ГБ на машине, **не меньше 4 ГБ для Docker** | пик тестов 2.1 ГБ |
| Диск | 6 ГБ свободно | образ 1.8 ГБ + кэш сборки; GPU-образ ещё 8.8 ГБ |
| GPU | не нужен | опционально: NVIDIA с драйвером ≥ 525 и поддержкой `docker run --gpus all`, см. ниже |
| ОС | Windows 10/11 с Docker Desktop (WSL2), Linux x86_64, macOS | на Apple Silicon образ работает через эмуляцию x86_64 и заметно медленнее |
| Docker | 23+ (проверено на 28.5.1) | нужен BuildKit, он включён по умолчанию с версии 23 |
| Git | любой | `.gitattributes` сам сохраняет LF в `*.sh` |

### Сеть

Интернет нужен **только при сборке образа**. Тесты и демо запускаются с
`--network none`; модель грузится из файла внутри образа, а тест
`test_model_loads_without_network` дополнительно блокирует сокеты и `torch.hub`.

Домены, к которым обращается сборка:

| Домен | Зачем |
|---|---|
| `registry-1.docker.io`, `auth.docker.io`, `production.cloudflare.docker.com` | базовый образ `python:3.10.15-slim-bookworm` |
| `download.pytorch.org` | колёса `torch` / `torchvision` (CPU или CUDA) |
| `pypi.org`, `files.pythonhosted.org` | остальные Python-пакеты |
| `huggingface.co`, `*.hf.co` (например `us.aws.cdn.hf.co`) | веса `scaled_offline.pth` (основной источник) |
| `www.modelscope.cn`, `cdn-lfs-cn-1.modelscope.cn` | те же веса, запасной источник (если CDN Hugging Face недоступен) |

Если недоступны оба источника весов, файл можно положить в `checkpoints/` вручную
(см. [checkpoints/README.md](checkpoints/README.md)): размер и sha256 проверяются в любом случае.

## Замеры

Машина: AMD Ryzen 7 7800X3D, Windows 10 + Docker Desktop 28.5.1 (WSL2), интернет ~80 Мбит/с.
Проверка с чистого клона: образы проекта и базовый `python` удалены, сборка с `--no-cache`,
папка `checkpoints/` пустая. Тесты — с ограничениями целевой машины `--cpus 4 --memory 8g`.

| Что | Результат |
|---|---|
| Сборка образа с нуля (`docker build --no-cache`, включая скачивание базового образа) | **109 с** |
| — из них веса: Hugging Face недоступен (43 с ожидания), затем ModelScope (24 с) | 67 с, параллельно с `pip install` (67 с) |
| Тесты при `--cpus 4 --memory 8g` | **101 с**, 13 passed |
| Тесты через `bash run.sh` без ограничений (8 потоков) | 74 с |
| **Итого: клон → сборка → тесты на 4 ядрах** | **≈ 3.5 мин** (бюджет 15 мин) |
| Размер образа `cotracker3:test` / `cotracker3:runtime` | 1.81 ГБ / 1.80 ГБ |
| Пиковая память при тестах (`docker stats`) | 2.06 ГБ |
| Пиковая загрузка CPU | 384% (все 4 ядра) |
| Демо на `apple.mp4` (50 кадров 1296×720, сетка 10×10) | инференс 20 с, всего 27 с, пик памяти 2.3 ГБ |
| Сеть в контейнере при `--network none` | нет (`socket.gaierror`) |

Время сборки почти целиком уходит на загрузки (PyTorch ~195 МБ, веса ~102 МБ); на
медленном канале оно растёт пропорционально. Если Hugging Face доступен, веса
скачиваются сразу, без 43 секунд ожидания.

## Ручная проверка по шагам

То же, что делает `run.sh`, но отдельными командами (в PowerShell замените `$(pwd)` на `${PWD}`):

```bash
docker build --target test -t cotracker3:test .
docker run --rm --network none -v "$(pwd)/results:/app/results" cotracker3:test
docker images cotracker3
```

Проверить, что в контейнере действительно нет сети:

```bash
docker run --rm --network none cotracker3:test python -c "import socket; socket.create_connection(('pypi.org', 443), 3)"
```

Ожидается ошибка `socket.gaierror` (имя не разрешается).

Пиковая память: во время тестов во втором терминале выполните `docker stats`.

Демо на своём видео:

```bash
bash run.sh --demo path/to/video.mp4 -- --grid_size 15 --resize 480
```

Результаты появятся в `results/demo/<имя видео>/`.

## Ожидаемый вывод pytest

```
============================= test session starts ==============================
platform linux -- Python 3.10.15, pytest-8.3.3, pluggy-1.7.0 -- /usr/local/bin/python
rootdir: /app
configfile: pytest.ini
testpaths: tests
collecting ... collected 13 items

tests/test_author_tests.py::test_author_bilinear_sampler PASSED          [  7%]
tests/test_e2e.py::test_demo_on_real_video PASSED                        [ 15%]
tests/test_e2e.py::test_procedural_video_accuracy[shapes] PASSED         [ 23%]
tests/test_e2e.py::test_procedural_video_accuracy[pan_zoom] PASSED       [ 30%]
tests/test_model_load.py::test_checkpoint_is_local PASSED                [ 38%]
tests/test_model_load.py::test_model_loads_without_network PASSED        [ 46%]
tests/test_outputs.py::test_output_shapes_and_types PASSED               [ 53%]
tests/test_outputs.py::test_output_values_finite_and_in_frame PASSED     [ 61%]
tests/test_reference.py::test_matches_reference PASSED                   [ 69%]
tests/test_synthetic.py::test_static_video PASSED                        [ 76%]
tests/test_synthetic.py::test_shift_accuracy[dx3_dy2] PASSED             [ 84%]
tests/test_synthetic.py::test_shift_accuracy[dx-2_dy4] PASSED            [ 92%]
tests/test_synthetic.py::test_occlusion PASSED                           [100%]

============================= CoTracker3: results ==============================
PASSED   test_author_bilinear_sampler              unittest of the authors: OK
                                                    -> results/tests/author_test_bilinear_sample.log
PASSED   test_demo_on_real_video                   50 frames -> mp4 50 frames 1296x720, gif, png, npy, json; visible 75.5%, inference 20.3 s
                                                    -> results/tests/e2e_apple
PASSED   test_procedural_video_accuracy[shapes]    error mean 0.155 px, p95 0.390 px (limit 1.5); visibility 99.7%, hidden points detected 100%
                                                    -> results/tests/e2e_shapes/gt_vs_pred.png
PASSED   test_procedural_video_accuracy[pan_zoom]  error mean 0.213 px, p95 0.428 px (limit 1.5); visibility 99.8%, hidden points detected 100%
                                                    -> results/tests/e2e_pan_zoom/gt_vs_pred.png
PASSED   test_checkpoint_is_local                  /app/checkpoints/scaled_offline.pth: 101890938 bytes, sha256 2670d4562ed6... OK
PASSED   test_model_loads_without_network          CoTrackerThreeOffline, 25.4M params, 188 tensors match the file, loaded in 0.2 s, network blocked
                                                    -> results/tests/model_info.json
PASSED   test_output_shapes_and_types              tracks (1, 24, 64, 2) torch.float32, visibility (1, 24, 64) torch.bool
PASSED   test_output_values_finite_and_in_frame    all values finite; visible points inside frame (+/-2 px); 404 predictions outside frame, all invisible
                                                    -> results/tests/outputs_grid.png
PASSED   test_matches_reference                    mean diff 0.0000 px, max 0.0000 px (limit 0.5); visibility agreement 100.00%
                                                    -> results/tests/reference_compare.png
PASSED   test_static_video                         drift mean 0.047 px, max 0.195 px (limit 1); visible 100.0%
                                                    -> results/tests/static_video.png
PASSED   test_shift_accuracy[dx3_dy2]              error mean 0.117 px, p95 0.229 px (limit 1); velocity (3.003, 2.000) vs (3, 2); visibility 99.0%
                                                    -> results/tests/shift_dx3_dy2.png
PASSED   test_shift_accuracy[dx-2_dy4]             error mean 0.091 px, p95 0.190 px (limit 1); velocity (-1.999, 4.000) vs (-2, 4); visibility 99.2%
                                                    -> results/tests/shift_dx-2_dy4.png
PASSED   test_occlusion                            16 covered points: 100% predicted hidden; others visible 100.0%; after the square: visible 100%, error 0.063 px
                                                    -> results/tests/occlusion.png
Visual results: results/tests/
============================= slowest 10 durations =============================
26.69s setup    tests/test_e2e.py::test_demo_on_real_video
13.83s call     tests/test_e2e.py::test_procedural_video_accuracy[shapes]
12.78s call     tests/test_e2e.py::test_procedural_video_accuracy[pan_zoom]
9.49s call     tests/test_synthetic.py::test_shift_accuracy[dx-2_dy4]
9.42s call     tests/test_synthetic.py::test_shift_accuracy[dx3_dy2]
9.40s call     tests/test_synthetic.py::test_occlusion
8.31s call     tests/test_synthetic.py::test_static_video
8.04s setup    tests/test_outputs.py::test_output_shapes_and_types
1.05s call     tests/test_author_tests.py::test_author_bilinear_sampler
0.88s call     tests/test_e2e.py::test_demo_on_real_video
======================== 13 passed in 101.20s (0:01:41) ========================
```

После pytest `run.sh` печатает список файлов в `./results/tests`, время и `TESTS DONE!`.

Время отдельных тестов на другой машине будет отличаться; метрики точности —
нет (на той же машине повторный прогон даёт побитово одинаковые треки).

## GPU (опционально)

```bash
bash run.sh --gpu
```

Собирается отдельный образ `cotracker3-gpu:test` (тот же `Dockerfile` с
`--build-arg REQUIREMENTS=requirements-gpu.txt`, PyTorch 2.4.1 + CUDA 12.1) и
запускается с `--gpus all -e COTRACKER_DEVICE=cuda`. CUDA входит в образ, на хосте
нужен только драйвер NVIDIA:

- Windows: Docker Desktop с WSL2 и свежий драйвер NVIDIA (отдельно ставить CUDA не нужно);
- Linux: драйвер NVIDIA и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Проверка, что Docker видит GPU:

```bash
docker run --rm --gpus all python:3.10.15-slim-bookworm nvidia-smi
```

Замер на GeForce GTX 1650 SUPER (4 ГБ): сборка GPU-образа 501 с (CUDA-библиотеки ~3 ГБ
загрузок), размер образа 8.84 ГБ, тесты — 13 passed за 29 с. Треки на GPU совпадают с
CPU-эталоном с точностью 0.0002 px.

## Устранение неполадок

### Docker Desktop не запускается, `docker info` с ошибкой (Windows, WSL2)

- Включите виртуализацию в BIOS/UEFI (Intel VT-x / AMD-V).
- В PowerShell от администратора: `wsl --install` (первый раз) или `wsl --update`, затем перезагрузка.
- Docker Desktop → Settings → General → включено **Use the WSL 2 based engine**.
- `run.sh` и `run.ps1` пишут `Docker daemon is not running`, если Docker Desktop не запущен.

### Нехватка памяти: `exit code 137`, `Killed`, контейнер падает на тестах

Docker Desktop на WSL2 по умолчанию даёт виртуальной машине половину RAM (на машине
с 8 ГБ это 4 ГБ). Тестам нужно около 2.1 ГБ, но если параллельно открыто много
программ, памяти может не хватить. Создайте `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=6GB
processors=4
```

затем выполните `wsl --shutdown` и перезапустите Docker Desktop. На macOS и на Windows
с Hyper-V: Docker Desktop → Settings → Resources → Memory. Для своего длинного видео
уменьшите нагрузку: `-- --max_frames 60 --resize 480 --fmaps_chunk 4`.

### `$'\r': command not found` или `bad interpreter` при `bash run.sh`

Скрипт получил окончания строк Windows (CRLF). `.gitattributes` в репозитории это
предотвращает, но если клон сделан со старыми настройками или файл копировали вручную:

```bash
git config core.autocrlf input
git rm --cached -r -q . && git reset --hard
```

или просто `sed -i 's/\r$//' run.sh`.

### Linux: `Permission denied` при записи в `results/` или файлы принадлежат root

`run.sh` запускает контейнер от текущего пользователя (`--user $(id -u):$(id -g)`) и
заранее создаёт `./results`. Если папку раньше создал Docker от root (например, при
ручном `docker run` или запуске через `sudo`), верните её себе:

```bash
sudo chown -R "$(id -u):$(id -g)" results reference
```

Не запускайте `run.sh` через `sudo`; вместо этого добавьте пользователя в группу
`docker` (`sudo usermod -aG docker $USER`, затем перелогиньтесь).

### Сборка падает на скачивании весов: `[weights] download failed`

Недоступны и CDN Hugging Face, и зеркало ModelScope. Скачайте `scaled_offline.pth`
любым способом, положите в `checkpoints/` и выполните `bash run.sh --rebuild`
(подробно в [checkpoints/README.md](checkpoints/README.md)).

### Сборка падает в `pip install`: `THESE PACKAGES DO NOT MATCH THE HASHES`, таймауты

Оборвалась загрузка большого колеса (на GPU-сборке случалось с CUDA-библиотеками).
Повторите `bash run.sh --rebuild`: готовые слои берутся из кэша. За прокси передайте
его при сборке: `docker build --build-arg HTTPS_PROXY=http://proxy:port ...`.

### PowerShell: `running scripts is disabled on this system`

Запустите один раз без изменения настроек системы:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### Git Bash: пути вида `C:/Program Files/Git/app/...`

`run.sh` сам отключает преобразование путей (`MSYS_NO_PATHCONV=1`). При ручных
командах `docker run` в Git Bash выполните `export MSYS_NO_PATHCONV=1`.

### `test_matches_reference` падает на другом компьютере

Эталон в `reference/` посчитан на AMD Ryzen 7 7800X3D. На другом процессоре числа
могут немного отличаться; допуск 0.5 px с запасом покрывает такую разницу. Если
расхождение больше, а остальные тесты точности проходят, пересоздайте эталон на своей
машине: `bash run.sh --make-reference`, затем `bash run.sh`.

### GPU: `could not select device driver "" with capabilities: [[gpu]]`

Docker не видит видеокарту: на Linux установите NVIDIA Container Toolkit и
перезапустите Docker, на Windows обновите драйвер NVIDIA и Docker Desktop.
Без GPU всё работает на CPU — просто запускайте без `--gpu`.
