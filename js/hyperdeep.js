// ------------------------------------------------------------------------------
// Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
// ACL 2023 - Supplementary Material
// Created on 18 jan. 2023
// ------------------------------------------------------------------------------

/* ----------------------------
 * PARAMETERS
 * ----------------------------*/
var channel_types = ["Forme", "Part-of-speech", "Lemme"];
var classifier;
var sorted_classifier;
var selected_class = 0;
var config = {};
var key_sentence_list;
var keysentence_limit = 20
var current_keysentence;
var current_page;
var global_max_TDS;
var global_max_ATT;
var sentences;

/* ----------------------------
 * PREDICT HANDLER
 * ----------------------------*/ 
function predictHandler(data) {

	// Setup config
	sentences = data["sentences"];
	config = data["config"];
	key_sentence_list = data["key_sentence_list"];
	classifier = data["classifier"];

	// Get a sorted classifier
	sortedData = Object.keys(classifier["classe-value"]).sort(function(a, b) {
	    return (classifier["classe-value"][a] > classifier["classe-value"][b]) ? -1 : 1;
	});
	sorted_classifier = [];
	$.each(sortedData, function(i, classe){
	    sorted_classifier.push([classe, classifier["classe-value"][classe]]);
	});

	selected_class = classifier["selected_class"];
	global_max_TDS = data["global_max_TDS"];
	global_max_ATT = data["global_max_ATT"];

	// Call hyperdeep visualisation library
	set_selected_class(selected_class);

}

/* ----------------------------
 * FULL TEXT CONTROLLER
 * ----------------------------*/ 
function show_sentences(page=current_page, skip_line=0) {

	$(".deconv_word").removeClass("faded-out");

	// SHOW SENTENCE
	current_page = page;
	html = '<p>';
	cpt = 1;
	$.each(sentences, function( i, sentence ) {
		if (i%2 != skip_line) return true;
		if (cpt <= page*keysentence_limit) {cpt++; return true};
		if (cpt > (page+1)*keysentence_limit) return false;

		$.each(sentence["html"], function(j, word ) {
			if (word.includes("__PARA__")) {
				html += word + "</p><p>";
			} else {
				html += word + " ";
			}
		});
		cpt++;
	});
	html += "</p>";
	$("#sentences").html(html);

	// NEXT SENTENCE BUTTON
	next_page_button = '<div class="hyperdeep-nav" style="text-align: center; position: relative">';
	if(page > 0) {
		next_page_button += '<a href="#" onclick="show_sentences(' + (page-1) + ');">◀︎ précédent</a>';
	} else {
		next_page_button += '<a style="color: grey;">◀︎ précédent</a>';
	}
	next_page_button += ' | ' + (page+1) + ' | ';
	if(page+1 < ((sentences.length*2)/(keysentence_limit*3))-1) {
		next_page_button += '<a href="#" onclick="show_sentences(' + (page+1) + ');">suivant ►</a>';
	} else {
		next_page_button += '<a style="color: grey;">suivant ►</a>';
	}
	next_page_button += '</div><br /><br />';
	$("#next_page").html(next_page_button);

	// UPDATE TDS UI
	update_TDS_markers("#sentences");

}


function focus_on_keysentence(keysentence=current_keysentence) {
	
	// GET KEY-SENTENCE-ID
	current_keysentence = keysentence;	
	keysentence_id = "key-sentence-" + key_sentence_list[selected_class][keysentence];

	// GO TO KEY-SENTENCE PAGE
	current_page = Math.floor(key_sentence_list[selected_class][keysentence]/(keysentence_limit*2));
	show_sentences(current_page, (key_sentence_list[selected_class][keysentence])%2);

	// SHOW ONLY KEY-SENTENCE IN TEXT
	$(".deconv_word").removeClass("faded-out");
	$(".deconv_word:not(." + keysentence_id + ")").addClass("faded-out"); 
	
	// SCROLL VIEW
	setTimeout(function(){
		next_button = document.getElementById(keysentence_id);
		next_button.scrollIntoView({behavior: "smooth"});
	}, 100);

}

function toogle_focus() {

	if ($( ".faded-out" ).length) {
		$(".deconv_word").removeClass("faded-out");
	} else {
		focus_on_keysentence();
	}

}

/* ----------------------------
 * KEYSENTENCE CONTROLLER
 * ----------------------------*/ 
function show_keysentences(keysentence=current_keysentence) {

	current_keysentence = keysentence;
	$(".deconv_word").removeClass("faded-out");

	// Print key-sentence 
	$("#key-sentences").empty();
	current_keysentence_html = sentences[key_sentence_list[selected_class][keysentence]]["html"];
	$(current_keysentence_html).each(function(j, word){
		if (j == 0) return true; // Drop anchor
		$("#key-sentences").append(this);
		$("#key-sentences").append(" ");
		
	});
	$("#key-sentences").html("<p><span class='inter'>[...]</span> " + $("#key-sentences").html() + " <span class='inter'>[...]</span></p>");

	// UPDATE TDS UI
	update_TDS_markers("#key-sentences", selected_class);

	next_keysentence = '<div class="small-3 columns hyperdeep-nav" style="padding: 0px; text-align: left;">';
	next_keysentence += '<a href="#" onclick="toogle_focus();">voir dans le texte <i class="fi-magnifying-glass"></i></a>';
	next_keysentence += '</div>';
	next_keysentence += '<div class="small-6 columns hyperdeep-nav" style="text-align: center; position: relative">';
	if(keysentence > 0) {
		next_keysentence += '<a href="#" onclick="show_keysentences(' + (keysentence-1) + ');">◀︎ précédent</a>';
	} else {
		next_keysentence += '<a style="color: grey;">◀︎ précédent</a>';
	}
	next_keysentence += ' | ' + (keysentence+1) + ' | ';
	if(keysentence+1 < key_sentence_list[selected_class].length) {
		next_keysentence += '<a href="#" onclick="show_keysentences(' + (keysentence+1) + ');">suivant ►</a>';
	} else {
		next_keysentence += '<a style="color: grey;">suivant ►</a>';
	}
	next_keysentence += '</div>';
	next_keysentence += '<div class="small-3 columns hyperdeep-nav" style="padding: 0px; text-align: right;">';
	next_keysentence += "<a href='#' onclick='";
	next_keysentence += "$(\"#graph-modal\").foundation(\"reveal\", \"open\");";
	next_keysentence += "setTimeout(function(){draw_multibar_char(" + sentences[key_sentence_list[selected_class][keysentence]]["graph"] + ");}, 500);";
	next_keysentence += "'><i class='fi-graph-bar'></i> plus d'infos</a>";
	next_keysentence += '</div>';
	$("#next_keysentence").html(next_keysentence);

}


function remove_anchors(sample) {
	result = []
	for (i = 0; i < sample.length; i++) { 
		if (!$($.parseHTML(sample[i])).hasClass("anchor")) {
			result.push(sample[i]);
		}
	}
	return result;
}

function get_keypassages(sample, from, key_passage={"words" : [], "score" : 0}, rec=0) {

	key_passages = [];

	att_threshold = parseFloat($("#myRangeATT").val());
	tds_thresholds = [];
	for (i = 0; i < 3; i++) { 
		tds_thresholds[i] = parseFloat($("#myRange" + i).val());
	}

	// --------------------------------
	// LOOP ON EACH WORD
	// --------------------------------
	links = [];
	sample = remove_anchors(sample);
	for (var wordindex1 = from; wordindex1 < sample.length; wordindex1++) { 

		if (key_passage["words"].includes(wordindex1)) {
			console.log("*****************************************************************************************************************************")
			console.log("index " + wordindex1 + "  already exist!");
			return key_passages;
		} 

		// Convert string to obj
		word1 = $.parseHTML(sample[wordindex1]);
			
		// Get word wTDS
		data_tds = JSON.parse($(word1).attr("data-tds"));
		word1_tds = 0;
		word1_channel = 0;
		for (channel = 0; channel < config["nb_channels"]; channel++) { 
			if (data_tds[channel][selected_class] > tds_thresholds[channel] && data_tds[channel][selected_class] > word1_tds) {
				word1_tds = data_tds[channel][selected_class];
				word1_channel = channel;
			}
		}
		
		if (word1_tds > 0) {
			console.log(wordindex1 + " => channel => " + word1_channel);

			//console.log(key_passage);
			key_passage["words"].push(wordindex1);
			key_passage["score"] += word1_tds;

			// Attention => Jump to 
			if ($(word1).attr("data-att") !== undefined) {
				data_att = JSON.parse($(word1).attr("data-att"));
				for (att = 0; att < data_att[word1_channel].length; att++) { 
					wordindex2 =  data_att[word1_channel][att][0]%config["SEQUENCE_SIZE"];
					att_score = data_att[word1_channel][att][1];
					console.log("test on " + wordindex2 + ":");
					console.log(att_score + " >= " + att_threshold);
					if (wordindex2 != wordindex1 && att_score >= att_threshold) {
						word2 = $.parseHTML(sample[wordindex2]);
						word2_channel = parseInt(data_att[word1_channel][att][0]/config["SEQUENCE_SIZE"]);
						data_tds = JSON.parse($(word2).attr("data-tds"));
						word2_tds = data_tds[word2_channel][selected_class];
						console.log(wordindex2 + " : " + word2_tds);
						if (word2_tds > tds_thresholds[word2_channel] || global_max_TDS == 0) {							
							link = {"dest" : wordindex2, "score" : att_score};
							console.log(link)
							links.push(link);
						}
					}
				}
			}
		} else {
			if (links.length > 0) {
				/*
				links.sort(function(a, b) {
					if (a["score"] < b["score"]) return 1;
					if (a["score"] > b["score"]) return -1;
					return 0;
				});
				*/
				for (l=0 ; l<links.length ; l++) {
					key_passage["score"] *= links[l]["score"]
					console.log("Jump to:");
					console.log(links[l]);
					console.log("with:");
					console.log(key_passage);
					if (rec == 0) {
						rec += 1
						key_passages = key_passages.concat(get_keypassages(sample, links[l]["dest"], key_passage, rec));					
					}
				}
			} else {
				//if (key_passage["words"].length > 0) {
				if (key_passage["words"].length > 0) {	
					console.log("break on: " + sample[wordindex1])
					key_passages.push(key_passage);
					key_passage = {"words" : [], "score" : 0}
				}
				//break;
			}
		}
	}

	return key_passages;
}

function show_keypassages() {

	console.log("--------------");
	console.log("show_keypassages");
	console.log("--------------");

	// Print key-passage
	$("#key-passages").empty();
	all_key_passages = [];
	key_sentences = key_sentence_list[selected_class].slice(0, keysentence_limit*2);

	// --------------------------------
	// LOOP ON EACH SAMPLE
	// --------------------------------
	$.each(key_sentences, function(e, i) {
	//for (i = 0; i < key_sentences.length; i++) { 
			
		//all_passage = []
		j=0
		//for (var j = 0; j < sentences[i]["html"].length; j++) { 
			//if (all_passage.includes(j)) continue;
			
			// get keypassages
			key_passages = get_keypassages(sentences[i]["html"], j);
			//console.log("key_passages:");
			$.each(key_passages, function(p, key_passage){
				//console.log(p);
				//console.log(key_passage);
				score = key_passage["score"];
				words = key_passage["words"].sort();
				//if (all_passage.includes(words[words.length-1])) continue;
				//all_passage = all_passage.concat(words);
				
				if (words.length > 0) {
					html = [];
					for (p=0 ; p<words.length ; p++) {
						if (p>0 && words[p-1] < words[p]-1) {
							html.push("<span>[...]</span>");
						}
						html.push(remove_anchors(sentences[i]["html"])[words[p]]);
					}
					all_key_passages.push({
						"key_sentence" : e,
						"score" : score.toFixed(2),
						"html" : html
					});
				}
			});
		//}
	});

	keys = Object.keys(all_key_passages);
	all_key_passages.sort(function(a, b){
		return b["score"]-a["score"];
	});

	var cpt = 0
	all_key_passages.forEach(function(entry) {
		tr = $("<tr class='select_item' onclick='show_keysentences(" + entry["key_sentence"] + "); toogle_focus();'></tr>");
		$(tr).append("<td>" + entry["html"].join(' ') + "</td>");
		$(tr).append("<td>" + entry["score"] + "</td>");
		$("#key-passages").append(tr);
		cpt++;
	});

	update_TDS_markers("#key-passages");
}

function update_TDS_markers(where=false, classe=selected_class) {

	if (where) {
		where = where + " .deconv_word";
	} else {
		where = " .deconv_word";
	}
	$(where).each(function( index ) {
		$(this).removeClass("TDS0 TDS1 TDS2");
		
		if ($(this).attr("data-tds")) {
			data_tds = JSON.parse($(this).attr("data-tds"));
		} else {
			data_tds = false;
		}
		data_str = JSON.parse($(this).attr("data-str"));
		already_highlighted = false;

		$(this).text(data_str[0]);

		//
		for (channel = 0; channel < config["nb_channels"]; channel++) { 
			tds = data_tds[channel][classe];
			str = data_str[channel];
			threshold = parseFloat($("#myRange" + channel).val());
			if (tds > threshold) {
				if (!already_highlighted || tds > already_highlighted) {
					$(this).text(str); 
					$(this).removeClass("TDS0 TDS1 TDS2");
					$(this).addClass("TDS" + channel);
				}
				already_highlighted = tds;
			}
		}
	});
	update_ATT_markers('#key-sentences');
}

function update_ATT_markers(where=false) {
	if (where) {
		where = where + " .deconv_word";
	} else {
		where = " .deconv_word";
	
	}
	threshold = parseFloat($("#myRangeATT").val());
	links = []
	$(where).each(function( wordindex1 ) {
		if ($(this).attr("data-att") !== undefined) {
			data_att = JSON.parse($(this).attr("data-att"));
			for (wordchannel1 = 0; wordchannel1 < config["nb_channels"]; wordchannel1++) {
				if ($(this).attr("class").indexOf("TDS"+wordchannel1) >= 0 || global_max_TDS == 0) {
					for (att = 0; att < data_att[wordchannel1].length; att++) { 
						att_score = data_att[wordchannel1][att][1];
						if (att_score >= threshold) {
							att_destination = data_att[wordchannel1][att][0];
							wordindex2 =  att_destination%config["SEQUENCE_SIZE"];
							wordchannel2 = parseInt(att_destination/config["SEQUENCE_SIZE"]);
							if ($($(where)[wordindex2]).attr("class").indexOf("TDS"+wordchannel2) >= 0 || global_max_TDS == 0) {								
								link = {"src" : $(where)[wordindex1], "dest" : $(where)[wordindex2], "score" : att_score}
								links.push(link)
							}
						}
					}
				}
			}
		}
	});
	links.sort(function(a, b) {
		if (a["score"] < b["score"]) return 1;
		if (a["score"] > b["score"]) return -1;
		return 0;
	});

	// DRAW LINES
	// distance of control point from mid-point of line:
	var offset = 30;
	$("#key-sentences-svg").css("width", ($("#key-sentences").css("width")));
	$("#key-sentences-svg").css("height", ($("#key-sentences").css("height") + (offset*2)));
	$("#key-sentences-svg").css("top", ($("#key-sentences").position().top-offset));

	curves = "";
	if (links.length > 0) {
		$(where).css("opacity", "0.2");
	}

	var opacity = 1.1
	$(links).each(function(l, link ) {
		$(link["src"]).css("opacity", "1");
		$(link["dest"]).css("opacity", "1");
		src_index = $(where).index(link["src"]);	
		dest_index = $(where).index(link["dest"]);
		$(where).each(function( wordindex ) {
			if (wordindex == src_index-1 || wordindex == dest_index-1 || wordindex == src_index+1 || wordindex == dest_index+1) {
				if ($($(where)[wordindex]).css("opacity") < 1) {
					$($(where)[wordindex]).css("opacity", "0.5");
				}
			}
		});
		p1x = parseInt($(link["src"]).offset().left + $(link["src"]).width()/2  - $("#key-sentences").offset().left);
		p1y = parseInt($(link["src"]).offset().top - $("#key-sentences").offset().top) + offset;
		p2x = parseInt($(link["dest"]).offset().left + $(link["dest"]).width()/2  - $("#key-sentences").offset().left);
		p2y = parseInt($(link["dest"]).offset().top - $("#key-sentences").offset().top) + offset;

		// invert point order
		if (p1x > p2x) {
			px_tmp = p1x;
			py_tmp = p1y;
			p1x = p2x;
			p1y = p2y;
			p2x = px_tmp;
			p2y = py_tmp;
		} else if (p1x == p2x && p1y == p2y) return true;
		//console.log(p1x + ", " + p1y + " - " + p2x + ", " + p2y); 

		// mid-point of line:
		var mpx = (p2x + p1x) * 0.5;
		var mpy = (p2y + p1y) * 0.5;

		// angle of perpendicular to line:
		var theta = Math.atan2(p2y - p1y, p2x - p1x) - Math.PI / 2;

		// location of control point:
		var c1x = mpx + offset * Math.cos(theta);
		var c1y = mpy + offset * Math.sin(theta);

		// construct the command to draw a quadratic curve
		//var curve = "M " + p1x + " " + p1y + " L " + p2x + " " + p2y;
	    var curve = "M" + p1x + " " + p1y + " Q " + c1x + " " + c1y + " " + p2x + " " + p2y;
		curves += '<path d="' + curve + '" fill="transparent" stroke="red" stroke-opacity="' + opacity + '"></path>';
		opacity -= 0.1;
	});
	if (curves == "") {
		$(where).css("opacity", "1");	
	}
	$("#key-sentences-svg").html(curves);
}

/* ----------------------------
 * UPDATE PREDICTIONS
 * ----------------------------*/ 
function quickView(classe) {

	console.log("quickView");

	// UPDATE GLOBAL VRAIABLES
	selected_class = classe;
	update_TDS_markers();
}

function update_predictions() {
	$("#prediction").empty();
	$("#predictions_list").empty();
	predictions_list = $("#predictions_list");
	$.each(sorted_classifier, function(i, entry) {
		key = entry[0];
		value = Number.parseFloat(entry[1]).toPrecision(2);
		class_id = classifier["classes"].indexOf(key);
		if (key.indexOf(":") != -1) {
			key_str = key.split(":")[1];
		} else {
			key_str = key;
		}
		if (key == classifier["classes"][selected_class]){
			$("#prediction").append(key_str + " " + value + "%");
		} else {
			tr = "<tr class='select_item' onmouseover='quickView(" + class_id + ");' onmouseout='quickView(" + selected_class + ");'";
			if (value != 0) {
				tr += " onclick='set_selected_class(" + class_id + ");'";

			}
			tr += "'></tr>";
			tr = $(tr);
			tr.append("<td>" + key_str + "</td>");
			tr.append("<td>" + value + "%</td>");
			predictions_list.append(tr);
		}
	});
}

/* ----------------------------
 * UPDATE TDS THRESHOLDS
 * ----------------------------*/ 
function update_thresolds() {
	// UPDATE TDS THRESHOLD SLIDERS
	$("#thresholds").empty();
	for (channel = 0; channel < config["nb_channels"]; channel++) {
		max_TDS = global_max_TDS[selected_class][channel];
		if (max_TDS == 0) {
			global_max_TDS = 0;
			break;
		}
		newRange = '<div class="threshold"><label class="TDS' + channel + '">' + channel_types[channel] + '</label>';
		newRange += '<input type="range" ';
		newRange += 'id="myRange' + channel + '" ';
		newRange += 'min="0" ';
		newRange += 'max="' + (max_TDS) +  '" ';
		newRange += 'value="'+ ((max_TDS)/2) + '" ';
		newRange += 'step="'+ (max_TDS/50) + '" ';
		newRange += 'class="slider" ';  
		newRange += 'onchange="update_TDS_markers(\'#sentences\'); ';
		newRange += 'update_TDS_markers(\'#key-sentences\'); ';
		newRange += 'show_keypassages();"/></div>';
		$("#thresholds").append(newRange);
	} 

	if (global_max_ATT > 0) {
		newRange = '<br /><div class="threshold"><label class="ATT">Attention</label>';
		newRange += '<input type="range" ';
		newRange += 'id="myRangeATT" ';
		newRange += 'min="0" ';
		newRange += 'max="' + (global_max_ATT) +  '" ';
		newRange += 'value="'+ ((global_max_ATT)/2) + '" ';
		newRange += 'step="'+ (global_max_ATT/50) + '" ';
		newRange += 'class="slider" ';  
		newRange += 'onchange="update_ATT_markers(\'#key-sentences\'); show_keypassages();"/></div>';
		$("#thresholds").append(newRange);
	}

	if ($("#thresholds").children().length == 0) {
		$("#thresholds-container").empty();
	}
}

function set_selected_class(classe) {

	// UPDATE GLOBAL VRAIABLES
	selected_class = classe;
	current_page = 0;
	current_keysentence = 0;

	update_thresolds()
	update_predictions();
	show_sentences();
	show_keysentences();
	show_keypassages();
}




