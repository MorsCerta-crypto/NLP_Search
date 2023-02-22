

var search_inp = document.getElementById("input-search");
var select_sug = document.getElementById("select-suggestion");



//////////////////////////////////////////////////////////////////

/*
    Listen to input Event by tipping some text into the search-field

    Get suggestions by sending a GET request to the server, which will response with some possible suggestions
*/

search_inp.addEventListener("input", function() {
        get_suggestions(this.value);

});

async function get_suggestions(search_text) {
    if (!search_text?.length || search_text.length < 5) {
        update_suggestion([[],[]], '')
        return;
    }
    let http = await HttpGet('/suggestions', { text: search_text });

    let result = JSON.parse(http)
    console.log(result)
    update_suggestion(result['results'], search_text)
}

function select_suggestion(selected_value, suggestions) {
    console.log(suggestions[selected_value])
    search_inp.value = suggestions[selected_value];
}

function update_suggestion(suggestions, s_text) {
    if ((!suggestions[0].length && !suggestions[1].length) && s_text?.length) return;
    select_sug.options.length = 0          // Clear all Options
    select_sug.value = -1
    let createOptions = (suggestions, s_text) => {
        const index = suggestions[0].length ? 0 : 1;
        let sug = [];
        if (suggestions[index]) {
            suggestions[index].forEach((val, idx, arr) => {
                let text = ""
                if (index == 0) { // Suggestion current Word + next word
                    text = s_text.substr(0, s_text.lastIndexOf(" ")).trim() + " " + val[0] + " " + val[1];
                } else if (index == 1) { // Suggestion next word
                    text = s_text.trim() + " " + val;
                }
                if (text) {
                    let temp = document.createElement("option");
                    temp.text = text;
                    temp.value = idx;
                    select_sug.add(temp);
                    sug.push(text);
                }
            });
        }
        return sug;
    };

    const suggs = createOptions(suggestions, s_text)

    select_sug.size = suggs.length > 7 ? 7 : (suggs.length > 1 ? suggs.length : 2);

    if (suggs.length <= 0)
        select_sug.setAttribute("hidden", true);
    else
        select_sug.removeAttribute("hidden");

    select_sug.addEventListener("change", function() {
        select_suggestion(this.value, suggs)
    });
}

//////////////////////////////////////////////////////////////////////

/*
    Called by changing search Type at search.html

    Update Naming + Url to Request after selecting different Search-Type
    Show Field Search Input by selecting Field Search-Type
*/

function onChangeSearchType(obj) {
    let btn = document.getElementById("button-searchtype");
    let inp = document.getElementById("input-searchtype");
    let field = document.getElementById("field-selection");
    btn.innerHTML = obj.innerHTML;
    inp.value = obj.innerHTML;

    if (inp.value == 'Feldsuche')
        field.removeAttribute("hidden");
    else
        field.setAttribute("hidden", true);


    // update Html-Form Url
    let mapping = { Volltext: '/full', Feldsuche: '/field', Struktur: '/parsed'}
    let form = document.getElementById("search-form");
    console.log('changing to ' + mapping[inp.value])
    form.action = mapping[inp.value] || '/full';
}

function onChangeFieldType(obj) {
    let field_value = document.getElementById("input-field")
    let field_btn = document.getElementById("field-selection-btn")
    field_value.value = obj.innerHTML
    field_btn.innerHTML = obj.innerHTML
}


/*
    Listen to Hide Modal Event to change some Content to Hidden
*/
var modal = document.getElementById("contentModal");
if (modal) {
    modal.addEventListener('hidden.bs.modal', function() {
        let title = document.getElementById("contentModalLabel");
        let body  = document.getElementById("contentModalBody");
        let depend= document.getElementById("dependencyModalBodyT");
        title.innerHTML = "";
        body.innerHTML  = "";
        depend.innerHTML = "";
    })
}


/*
    Called by clicking a specific search result to show details info at search_results.html

    Get detail info Content by Http request and show Modal
*/
async function onClickOpenDetailModal(obj) {
    let title = document.getElementById("contentModalLabel");
    let body = document.getElementById("contentModalBody");
    let titleStr = obj.dataset.title;

    // Delete for Search in judgement Numbers at start
    titleStr = titleStr.replace(/[\d .,:]*/, "")

    title.innerHTML = titleStr;

    let id = obj.dataset.id.split("_")[0];
    let details = await HttpGet('/id', { id: id });
    console.log(details)

    // choose a randome char that is not used in the string 
    let outStr = JSON.parse(details)['results'][0]['content'].replace("&#160;", "§");

    const replace_umlauts = (str) => {
        let new_string = str
        const mapping = { 'Ä' : '196', 'ä': '228', 'Ö': '214', 'ö': '246', 'Ü': '220', 'ü': '252', 'ß': '223', '§' : '167' }
        Object.keys(mapping).forEach((key, idx) =>  {
            new_string = new_string.replaceAll(key, '&#' + mapping[key] + ';')
        });
        return new_string
    }
    console.log("Search for: " + replace_umlauts(titleStr))
    let searchTitle = -1
    try {
        searchTitle = outStr.replaceAll("<em>", "").replaceAll("</em>", "").replaceAll("§", " ").search(replace_umlauts(titleStr))
    } catch(err) {
        console.log('ERROR : ' + replace_umlauts(titleStr))
    }

    if (searchTitle >= 0) { // mark Sentence in Judgement
        outStr = outStr.substr(0, searchTitle) + "<p style='background-color: orange;'>" + outStr.substr(searchTitle) + "</p>";
    }

    body.innerHTML = outStr.replace("§", "&#160;");

    // set content for Dependency Modal && Scale down SVG
    let dep = document.getElementById("dependencyModalBodyT");
    let dependancyStr = obj.dataset.dependency;
    const parser = new DOMParser();
    const dummy = parser.parseFromString(dependancyStr, 'text/html');

    const svg = dummy.getElementsByTagName("svg")[0]
    const w = svg.getAttribute('width')
    const h = svg.getAttribute('height')
    svg.setAttributeNS(null, 'viewBox', '0 0 ' + w + ' ' + h)
    svg.setAttribute('width', Number(w) * 0.6)
    svg.setAttribute('height', Number(h) * 0.6)


    //dep.innerHTML = obj.dataset.dependency;
    dep.appendChild(svg)
    dep.setAttribute("hidden", true);
}

/*
    Called by clicking on open Modal show dependency
*/
function onClickShowDependency(obj) {
    let dep = document.getElementById("dependencyModalBodyT");
    if (dep.hasAttribute("hidden")) {
        dep.removeAttribute("hidden")
    } else {
        dep.setAttribute("hidden", true);
    }
}

/*
    Handle Http Get Requests
*/
function HttpGet(url, args) {
    return new Promise(function(resolve, reject) {
        const xhr = new XMLHttpRequest();
        if (Object.keys(args).length) {
            url += '?'
            Object.keys(args).forEach((arg_key, idx) => {
                if (idx)
                    url += '&'
                url += arg_key + '=' + args[arg_key].replace(" ", "+");
            });
        }
        xhr.open('GET', url);
        xhr.onload = function() {
            if (this.status >= 200 && this.status < 300) {
                resolve(xhr.response)
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText,
                });
            }
        }
        xhr.onerror = function() {
            reject({
                status: this.status,
                statusText: xhr.statusText,
            });
        }
        xhr.send();
    });
}
