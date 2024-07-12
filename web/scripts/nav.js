
function previous() {
    console.log(train_active)
    if (train_active) {
        if (counter_train > 0) {
            counter_train -= 1
            console.log(counter_train)
        }
    } else {
        if (counter_test > 0) {
            counter_test -= 1
            console.log(counter_test)
        }
    }
    update_view()
}
function next() {
    console.log(train_active)
    if (train_active) {
        if (counter_train + 1 < n_train) {
            counter_train += 1
            console.log(counter_train)
        }
    } else {
        if (counter_test + 1 < n_test) {
            counter_test += 1
            console.log(counter_test)
        }
    }
    update_view()
}

function numchange() {
    var in_elt = document.getElementById("item_number")
    if (train_active) {
        counter_train = in_elt.value - 1
    } else {
        counter_test = in_elt.value - 1
    }
    update_view()
}

function switch_dataset() {
    var set = document.getElementById("dataset").value
    train_active = set === "train"
    console.log(set)
    console.log(train_active)
    update_view()
}

function name_to_elt(ds, name) {
    if(ds === "test") {
        var img_src = "../assets/test_imgs/" + name;
        return '<img class="viz" src="' + img_src + '"></img>'
    } else {
        var urls_list = ds == "train" ? train_urls : test_urls
        var out = '<span class="cannot">Please view image at: '
        if(name in urls_list) {
            var url = urls_list[name]
            out += '<a href="' + url + '" target="_blank">(link)</a>'
        } else {
            out += '(n/a)'
        }
        out += '</span>'
        return out
    }
}

function update_view() {
    var c = train_active ? counter_train : counter_test
    var c1 = 1 + c
    var c2 = train_active ? n_train : n_test
    // document.getElementById("c1").innerHTML = c1
    document.getElementById("c2").innerHTML = c2
    // document.getElementById("dsname").innerHTML = train_active ? "Waldo and Wenda" : "imSitu-HHI"

    var in_elt = document.getElementById("item_number")
    in_elt.value = c1
    in_elt.min = 1
    in_elt.max = train_active ? 100 : 1000
    

    var ds = train_active ? "train" : "test"

    var name = (train_active ? train_names : test_names)[c]

    var elt_img = document.getElementById("results_img")

    elt_img.innerHTML = name_to_elt(ds, name)
    // elt.innerHTML += '<br><br>'

    var elt = document.getElementById("results")
    elt.innerHTML = ''


    if (ds === "test") {
        var mygt = gt["test"][name]
        var res_base = test_res["base"][name]
        var res_ft = test_res["ft"][name]
        
        elt.innerHTML += '<b>GT: </b><br>' + mygt.replaceAll('=>', '→') + '<br><br>'

        elt.innerHTML += '<b>Predictions for CLIP-Large: </b><br>' + res_base.join(' → ') + '<br><br>'

        elt.innerHTML += '<b>Predictions for CLIP-Large + alignment (fine-tuning): </b><br>' + res_ft.join(' → ') + '<br><br>'

        

        // elt.innerHTML += '<b>Ground truth: </b>' + mygt + '<br><br>'

        // var res = (train_active ? train_res : test_res)[model][name]
        // var top = res[0]

        // console.log(res)

        // elt.innerHTML += '<b>Top prediction:</b> ' + top + '<br><br>'

        // if (res.length > 1) {

        //     elt.innerHTML += '<b>Other predictions (lower-scoring beams):</b>'
        //     elt.innerHTML += '<ol>'
        //     for (var i = 1; i < res.length; i++) {
        //         elt.innerHTML += '<li>' + res[i] + '</li>'
        //     }
        //     elt.innerHTML += '</ol>'
        // }
    } else {
        var pos = gt[ds]["positive"][name].replaceAll('=>', '→')
        var neg = gt[ds]["negative"][name].replaceAll('=>', '→')

        elt.innerHTML += '<b>Positive captions: </b>' + pos + '<br><br>'
        elt.innerHTML += '<b>Negative captions: </b>' + neg + '<br><br>'

        
    }
    

}

update_view()