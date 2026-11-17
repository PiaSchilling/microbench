# json.awk - parser for flat JSON objects (library, load with -f before the main script)
#
# Only what the contract allows is accepted: one object whose values are
# strings, numbers, true, false or null. Nested objects/arrays are an error.
# POSIX awk (works with gawk and mawk).
#
# jparse(s)  parses s. Returns 1 on success, 0 on error (message in J_err).
#            Result: J_n             number of members
#                    J_key[i]        key (i = 1..J_n, in input order)
#                    J_type[i]       "s" string, "n" number, "b" boolean, "z" null
#                    J_val[i]        raw JSON text of the value (strings include quotes)
#                    J_idx[key]      index of key
# jdecode(v) returns the content of a raw JSON string without quotes and escapes.
# jstr(s)    returns s as JSON string literal (with quotes, escaped).

function jparse(s,    c, k) {
    J_n = 0; J_err = ""
    split("", J_key); split("", J_type); split("", J_val); split("", J_idx)
    J_s = s; J_i = 1; J_len = length(s)

    _jws()
    if (_jc() != "{") return _jfail("expected '{'")
    J_i++; _jws()
    if (_jc() == "}") { J_i++; return _jend() }
    while (1) {
        if (_jc() != "\"") return _jfail("expected a string key")
        if (!_jstr()) return 0
        k = jdecode(J_tok)
        if (k in J_idx) return _jfail("duplicate key '" k "'")
        _jws()
        if (_jc() != ":") return _jfail("expected ':' after key '" k "'")
        J_i++; _jws()
        if (!_jvalue(k)) return 0
        J_n++; J_key[J_n] = k; J_type[J_n] = J_ttype; J_val[J_n] = J_tok; J_idx[k] = J_n
        _jws(); c = _jc()
        if (c == ",") { J_i++; _jws(); continue }
        if (c == "}") { J_i++; return _jend() }
        return _jfail("expected ',' or '}' after value of '" k "'")
    }
}

function _jend() {
    _jws()
    if (J_i <= J_len) return _jfail("unexpected characters after the object")
    return 1
}

function _jfail(m) { J_err = m " (at character " J_i ")"; return 0 }

function _jc() { return substr(J_s, J_i, 1) }

function _jws(    c) {
    while (J_i <= J_len) {
        c = substr(J_s, J_i, 1)
        if (c == " " || c == "\t" || c == "\r" || c == "\n") J_i++
        else break
    }
}

function _jstr(    c, st) {
    st = J_i; J_i++
    while (J_i <= J_len) {
        c = substr(J_s, J_i, 1)
        if (c == "\\") { J_i += 2; continue }
        if (c == "\"") { J_i++; J_tok = substr(J_s, st, J_i - st); return 1 }
        J_i++
    }
    return _jfail("unterminated string")
}

function _jvalue(k,    c, rest) {
    c = _jc()
    if (c == "\"") { J_ttype = "s"; return _jstr() }
    if (c == "{" || c == "[") return _jfail("value of '" k "' is an object/array; only flat JSON is allowed")
    rest = substr(J_s, J_i)
    if (match(rest, /^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?/)) {
        J_tok = substr(rest, 1, RLENGTH); J_i += RLENGTH; J_ttype = "n"; return 1
    }
    if (substr(rest, 1, 4) == "true")  { J_tok = "true";  J_i += 4; J_ttype = "b"; return 1 }
    if (substr(rest, 1, 5) == "false") { J_tok = "false"; J_i += 5; J_ttype = "b"; return 1 }
    if (substr(rest, 1, 4) == "null")  { J_tok = "null";  J_i += 4; J_ttype = "z"; return 1 }
    c = tolower(substr(rest, 1, 1))
    if (c == "n" || c == "i" || tolower(substr(rest, 1, 2)) == "-i")
        return _jfail("invalid value for '" k "' (nan/inf are not valid JSON numbers)")
    return _jfail("invalid or missing value for '" k "'")
}

function jdecode(v,    out, i, n, c) {
    v = substr(v, 2, length(v) - 2); out = ""; n = length(v)
    for (i = 1; i <= n; i++) {
        c = substr(v, i, 1)
        if (c == "\\" && i < n) {
            i++; c = substr(v, i, 1)
            if (c == "n") c = "\n"
            else if (c == "t") c = "\t"
            else if (c == "r") c = "\r"
            else if (c == "b" || c == "f") c = " "
            else if (c == "u") c = "\\u"     # kept as is
        }
        out = out c
    }
    return out
}

function jstr(s) {
    gsub(/\\/, "\\\\", s); gsub(/"/, "\\\"", s)
    gsub(/\t/, "\\t", s); gsub(/\r/, "\\r", s); gsub(/\n/, "\\n", s)
    return "\"" s "\""
}
