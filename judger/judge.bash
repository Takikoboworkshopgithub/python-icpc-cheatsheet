#!/bin/bash

# --- 設定項目 ---
SOURCE_DIR="src"                    # ソースコードがあるディレクトリ
TEST_DIR="tests"                    # テストケースがあるディレクトリ
TIME_LIMIT_SEC=2                    # 実行時間制限 (秒)
MEMORY_LIMIT_MB=2048                 # メモリ制限 (MB, C/C++のみ目安)

# --- 色定義 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- ヘルプ表示関数 ---
function show_help() {
    echo "Usage: judge <source_filename>"
    echo "Example: judge a.c"
    echo "Example: judge b.cpp"
    echo "Example: judge c.py"
    echo ""
    echo "Supported languages: C, C++, Python"
    echo "Source files are assumed to be in the '${SOURCE_DIR}/' directory."
    echo "Test files should match the source file prefix (e.g., c.py -> tests/c*.in, tests/c*.out)"
    exit 1
}

# --- メイン処理開始 ---

# 引数チェック
if [ -z "$1" ]; then
    echo -e "${RED}エラー: ソースファイル名が指定されていません。${NC}"
    show_help
fi

# 入力されたファイル名に自動的に src/ を付与
SOURCE_FILE="${SOURCE_DIR}/$1"
FILENAME=$(basename "$SOURCE_FILE")
EXTENSION="${FILENAME##*.}"
SOURCE_PREFIX="${FILENAME%.*}" # ソースファイルの頭文字（例: a.py -> a）

EXECUTABLE_NAME="a.out" # C/C++の場合の実行ファイル名

# 指定されたソースファイルが存在するかチェック
if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}エラー: ソースファイル '${SOURCE_FILE}' が見つかりません。${NC}"
    echo "指定されたファイルは '${SOURCE_DIR}/' ディレクトリに存在する必要があります。"
    exit 1
fi

echo -e "${BLUE}--- ジャッジ開始 ---${NC}"
echo -e "${BLUE}ソースファイル: ${SOURCE_FILE}${NC}"
echo -e "${BLUE}時間制限: ${TIME_LIMIT_SEC}秒, メモリ制限: ${MEMORY_LIMIT_MB}MB${NC}"
echo "--------------------"

# 既存の実行ファイルを削除
if [ -f "$EXECUTABLE_NAME" ]; then
    rm "$EXECUTABLE_NAME"
fi

# 一時ファイルのクリーンアップ関数
function cleanup() {
    rm -f temp_output.txt error_output.txt "$EXECUTABLE_NAME" compile_error.txt execution_metrics.txt
}
trap cleanup EXIT # スクリプト終了時にクリーンアップを実行

# --- コンパイルフェーズ (C/C++のみ) ---
COMPILE_REQUIRED=false
if [[ "$EXTENSION" == "c" ]]; then
    echo -e "${BLUE}--- C言語 コンパイル中 ---${NC}"
    gcc "$SOURCE_FILE" -o "$EXECUTABLE_NAME" -Wall -O2 2> "compile_error.txt"
    COMPILE_EXIT_CODE=$?
    COMPILE_REQUIRED=true
elif [[ "$EXTENSION" == "cpp" ]]; then
    echo -e "${BLUE}--- C++言語 コンパイル中 ---${NC}"
    g++ "$SOURCE_FILE" -o "$EXECUTABLE_NAME" -Wall -O2 -std=c++17 2> "compile_error.txt"
    COMPILE_EXIT_CODE=$?
    COMPILE_REQUIRED=true
fi

if [ "$COMPILE_REQUIRED" == "true" ]; then
    if [ $COMPILE_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}[CE] コンパイルエラーが発生しました！${NC}"
        echo -e "${RED}--- コンパイルエラー出力 ---${NC}"
        cat "compile_error.txt"
        echo -e "${RED}--------------------------${NC}"
        rm -f compile_error.txt # エラー出力後はファイルを削除
        exit 1
    else
        echo -e "${GREEN}コンパイル成功！${NC}"
        rm -f compile_error.txt # 成功時はファイルを削除
    fi
    echo ""
fi

# --- テストケース実行フェーズ ---
# ソースファイルの頭文字に一致するテストケースを検索
TEST_FILES=$(find "$TEST_DIR" -name "${SOURCE_PREFIX}*.in" | sort)

if [ -z "$TEST_FILES" ]; then
    echo -e "${YELLOW}警告: '${SOURCE_PREFIX}' で始まるテストケースが見つかりませんでした: ${TEST_DIR}/${SOURCE_PREFIX}*.in${NC}"
    exit 0
fi

for input_file in $TEST_FILES; do
    # .in を .out に置換して期待される出力ファイル名を生成
    expected_output_file="${input_file%.in}.out"
    test_name=$(basename "$input_file" .in)

    echo -e "${BLUE}--- テストケース: ${test_name} ---${NC}"

    if [ ! -f "$expected_output_file" ]; then
        echo -e "${YELLOW}警告: ${input_file} に対応する期待される出力ファイル (${expected_output_file}) が見つかりません。このテストケースはスキップされます。${NC}"
        echo ""
        continue
    fi

    # メモリ制限の設定 (ulimit -v はVMサイズなので、少し余裕を持たせる)
    # 実際のメモリ使用量を正確に取得するのは難しいので、これはあくまで目安です。
    # C/C++ のみで ulimit を適用し、Python は別の方法でエラー判定を試みる
    if [[ "$EXTENSION" == "c" || "$EXTENSION" == "cpp" ]]; then
        ulimit -v $((MEMORY_LIMIT_MB * 1024)) &> /dev/null
    fi

    # 実行時間とメモリ使用量を計測するために /usr/bin/time を使用
    # -f オプションでフォーマットを指定: %e (経過時間), %M (最大常駐セットサイズ)
    # 2> execution_metrics.txt は /usr/bin/time の出力をリダイレクト
    # その後ろの 2> error_output.txt は対象プログラムの標準エラー出力をリダイレクト
    EXEC_STATUS=0 # 初期化
    case "$EXTENSION" in
        c|cpp)
            /usr/bin/time -f "%e %M" -o execution_metrics.txt \
                timeout $TIME_LIMIT_SEC "./$EXECUTABLE_NAME" < "$input_file" > "temp_output.txt" 2> "error_output.txt"
            EXEC_STATUS=$?
            ;;
        py)
            # Python は -Xmx のような直接的なメモリ制限オプションがないため、ulimit が主な手段。
            # ただし、ulimit -v は仮想メモリなので、Pythonの実際のメモリ使用量を厳密に制限するのは難しい。
            # Python のメモリ計測は /usr/bin/time を使ってもRES(常駐セットサイズ)が正確に出にくい場合がある
            /usr/bin/time -f "%e %M" -o execution_metrics.txt \
                timeout $TIME_LIMIT_SEC python3 "$SOURCE_FILE" < "$input_file" > "temp_output.txt" 2> "error_output.txt"
            EXEC_STATUS=$?
            ;;
        *)
            echo -e "${RED}対応していないファイル拡張子です: ${EXTENSION}${NC}"
            continue
            ;;
    esac

    # ulimit の設定を元に戻す
    if [[ "$EXTENSION" == "c" || "$EXTENSION" == "cpp" ]]; then
        ulimit -v unlimited &> /dev/null
    fi

    # 実行メトリクスを読み込む
    ELAPSED_TIME="N/A"
    MAX_MEMORY_KB="N/A"
    if [ -f "execution_metrics.txt" ]; then
        read ELAPSED_TIME MAX_MEMORY_KB < execution_metrics.txt
        MAX_MEMORY_MB=$(echo "scale=2; $MAX_MEMORY_KB / 1024" | bc)
    fi

    if [ $EXEC_STATUS -eq 124 ]; then
        echo -e "${YELLOW}[TLE] 時間制限超過 (${TIME_LIMIT_SEC}秒)${NC}"
    elif [ $EXEC_STATUS -ne 0 ]; then
        # ランタイムエラーの判定
        IS_MLE=false
        if [ -s "error_output.txt" ]; then
            # C/C++ で ulimit によるメモリ制限超過の場合
            if [[ "$EXTENSION" == "c" || "$EXTENSION" == "cpp" ]] && grep -q "memory limit exceeded" "error_output.txt"; then
                IS_MLE=true
            fi
            # Python の場合は、OSError: [Errno 12] Cannot allocate memory や MemoryError を検知
            if [[ "$EXTENSION" == "py" ]] && grep -q -E "(MemoryError|Cannot allocate memory)" "error_output.txt"; then
                IS_MLE=true
            fi
        fi

        if [ "$IS_MLE" == "true" ]; then
             echo -e "${YELLOW}[MLE] メモリ制限超過 (${MEMORY_LIMIT_MB}MB)${NC}"
        else
            echo -e "${RED}[RE] ランタイムエラー (終了コード: $EXEC_STATUS)${NC}"
            if [ -s "error_output.txt" ]; then
                echo -e "${RED}--- エラー出力 ---${NC}"
                cat "error_output.txt"
                echo -e "${RED}------------------${NC}"
            fi
        fi
    else
        # WA (Wrong Answer) または AC (Accepted) チェック
        # diff -Z: 行末の空白と改行コードを無視して比較
        if ! diff -qZ "temp_output.txt" "$expected_output_file" > /dev/null 2>&1; then
            echo -e "${RED}[WA] 不正解！${NC}"
            echo -e "${RED}--- あなたの出力 ---${NC}"
            cat "temp_output.txt"
            echo -e "${RED}--- 期待される出力 ---${NC}"
            cat "$expected_output_file"
            echo -e "${RED}--------------------${NC}"
        else
            echo -e "${GREEN}[AC] 正解！${NC}"
        fi
    fi
    echo -e "${BLUE}実行時間: ${ELAPSED_TIME}秒, メモリ: ${MAX_MEMORY_MB}MB${NC}" # 実行時間とメモリも出力
    echo "" # 各テストケースの区切り
done

echo -e "${BLUE}--- ジャッジ完了 ---${NC}"