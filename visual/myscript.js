function remove_pfx(_exp) {
    _wexp = []
    for (const x of _exp) {
        _g = x[0].replace(/^\d+__/, '')
        _words = _g.split('__')
        for (const _w of _words) {
            _wexp.push([_w, 0, x[1]])
        }
    }
    _exp = _wexp

    let _dict = {}
    for (const x of _exp) {
        // mt = x[0].match(/^\d_\d+_(\d+)_/)
        mt = x[0].match(/^(\d+)_/)
        _dict[mt[1]] = x
    }

    _dict = Object.keys(_dict)
        .sort()
        .reduce((accumulator, key) => {
            accumulator[key] = _dict[key];
            return accumulator;
        }, {});

    _sep = 0
    _keys = Object.keys(_dict)
    for (let i = 0; i < _keys.length; i++) {
        _keys[i] = parseInt(_keys[i])
    }
    for (let i = 0; i < _keys.length - 1; i++) {
        if (_keys[i + 1] - _keys[i] > 1) {
            _sep = _keys[i] + 1
        }
    }

    _exp = []
    let i = 0
    let tot_len = 0
    _raw_text = ''
    for (const k of Object.keys(_dict)) {
        let x = _dict[k]
        let word = x[0]
        // x[0] = word.replace(/^\d_\d+_\d+_/, '')
        x[0] = word.replace(/^\d+_/, '')
        x[1] = tot_len
        tot_len += x[0].length + 1
        _exp[i++] = x
        if (parseInt(k) == _sep + 1) {
            _raw_text += '\n'
        } else {
            _raw_text += ' '
        }
        _raw_text += x[0]
    }

    return [_exp, _raw_text.substring(1)]
}


function parse_groups(_exp) {
    _wexp = []
    for (const x of _exp) {
        _g = x[0].replace(/^\d+__/, '')
        _words = _g.split('__')
        group = []
        for (const _w of _words) {
            // group.push([_w.replaceAll(/\d_\d+_\d+_/g, ''), 0, x[1]])
            group.push([_w.replaceAll(/\d+_/g, ''), 0, x[1]])
        }
        _wexp.push(group)
    }
    console.log(_wexp)

    _exp = []
    let i = 0
    let tot_len = 0
    let _raw_text = ''
    for (const g of _wexp) {
        for (const x of g) {
            _exp.push([x[0], tot_len, x[2]])
            tot_len += x[0].length + 1
            _raw_text += x[0]
            _raw_text += ' '
        }
        _raw_text += '\n'
    }
    _raw_text = _raw_text.replaceAll(' \n', '\n')

    return [_exp, _raw_text]
}

function visual() {
    var top_div = d3.select('#top_div54C11HAMC3G6B7T').classed('lime top_div', true);
    var pp_div = top_div.append('div').classed('lime predict_proba', true);
    var pp_svg = pp_div.append('svg').style('width', '100%');
    var pp = new lime.PredictProba(pp_svg, ["no_match", "match"], predict_proba);

    var exp_div;
    var exp = new lime.Explanation(["no_match", "match"]);
    exp_div = top_div.append('div').classed('lime explanation', true);

    // // let _top_words_coefs = [["3__11_21__24_02", -0.4827866918630964], ["6__9_27860__22_27860", 0.39657485211429677], ["5__8_imation__21_imation", 0.19929218463807277], ["0__6_usb__7_drives__18_car__19_audio__20_video", -0.1892728694549917], ["1__2_32gb__4_flash__5_drive__14_32gb__16_flash__17_drive", -0.15788059706442617], ["2__1_imation__3_pocket__13_usb__15_pocket", 0.14077907206747017], ["4__10_77__23_44", -0.05970107062982683]]
    // const top_words_coefs = []
    // for (const x of _top_words_coefs) {
    //     // let g = x[0].replace(/^\d+__/, '').replaceAll(/\d_\d+_\d+_/g, '').replaceAll('__', '_')
    //     let g = x[0].replace(/^\d+__/, '').split('__')
    //     for (const i in g) {
    //         g[i] = g[i].replace(/^\d+_/g, '')
    //     }
    //     g = g.join('_')
    //     top_words_coefs.push([g, x[1]])
    // }
    // exp.show(top_words_coefs, 1, exp_div);
    exp.show(_top_words_coefs, 1, exp_div);

    var raw_div = top_div.append('div');

    // let _words_coefs = [["6__9_27860__22_27860", 0.39657485211429677], ["5__8_imation__21_imation", 0.19929218463807277], ["2__1_imation__3_pocket__13_usb__15_pocket", 0.14077907206747017], ["4__10_77__23_44", -0.05970107062982683], ["1__2_32gb__4_flash__5_drive__14_32gb__16_flash__17_drive", -0.15788059706442617], ["0__6_usb__7_drives__18_car__19_audio__20_video", -0.1892728694549917], ["3__11_21__24_02", -0.4827866918630964]]
    let [words_coefs, raw_text_] = remove_pfx(_words_coefs)
    let [gr_words_coefs, gr_raw_text] = parse_groups(_words_coefs)
    for (let i in gr_words_coefs) {
        gr_words_coefs[i][1] += (raw_text_.length + 2)
    }
    words_coefs = words_coefs.concat(gr_words_coefs)
    raw_text_ += '\n\n' + gr_raw_text
    exp.show_raw_text(words_coefs, 1, raw_text_, raw_div, true);
}
