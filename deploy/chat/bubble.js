/* A drop-in chat bubble that mounts LibreChat in the corner of any page.
 *
 *   <script src="/chat/bubble.js" data-chat-url="http://localhost:3080"></script>
 *
 * Deliberately unstyled beyond what it takes to work. This is the integration, not the design:
 * it owns the iframe lifecycle, the toggle, and the constraints below, and leaves every visual
 * decision to whatever stylesheet the host page brings. Override by targeting the two classes.
 *
 * Three constraints worth knowing before wiring this into a page.
 *
 * Serve the host page over http://localhost, not file://. LibreChat keeps a session cookie, and
 * a page opened from the filesystem makes that cookie third-party, which Chrome drops by
 * default. The symptom is a login screen that reappears on every reload, and it looks like a
 * LibreChat bug rather than an embedding mistake. Ports do not matter -- cookies ignore them --
 * so any localhost port is fine.
 *
 * The iframe is created on first open, not on load. LibreChat is a full application and booting
 * one inside every page view to serve the readers who never click would be rude to the page it
 * is a guest in.
 *
 * There is no cross-frame messaging. Sending the current case into the conversation would need
 * postMessage on both sides and a LibreChat that listens, which it does not. Anyone asking a
 * follow-up types it, and the model reaches the same warehouse the page was rendered from.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var url = (script && script.dataset.chatUrl) || "http://localhost:3080";
  var label = (script && script.dataset.label) || "Ask about this data";

  var style = document.createElement("style");
  style.textContent = [
    ".verdict-chat-toggle{position:fixed;right:20px;bottom:20px;z-index:2147483000;",
    "  border:0;border-radius:999px;padding:12px 18px;font:600 14px/1 system-ui,sans-serif;",
    "  background:#2f81f7;color:#fff;cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,.3)}",
    ".verdict-chat-panel{position:fixed;right:20px;bottom:76px;z-index:2147483000;",
    "  width:min(420px,calc(100vw - 40px));height:min(620px,calc(100vh - 120px));",
    "  border:1px solid rgba(128,128,128,.35);border-radius:12px;overflow:hidden;",
    "  background:#fff;box-shadow:0 12px 40px rgba(0,0,0,.35);display:none}",
    ".verdict-chat-panel.open{display:block}",
    ".verdict-chat-panel iframe{width:100%;height:100%;border:0;display:block}",
  ].join("");
  document.head.appendChild(style);

  var panel = document.createElement("div");
  panel.className = "verdict-chat-panel";

  var toggle = document.createElement("button");
  toggle.className = "verdict-chat-toggle";
  toggle.type = "button";
  toggle.textContent = label;
  toggle.setAttribute("aria-expanded", "false");

  toggle.addEventListener("click", function () {
    var opening = !panel.classList.contains("open");
    if (opening && !panel.firstChild) {
      var frame = document.createElement("iframe");
      frame.src = url;
      frame.title = "Verdict chat";
      frame.allow = "clipboard-write";
      panel.appendChild(frame);
    }
    panel.classList.toggle("open", opening);
    toggle.setAttribute("aria-expanded", String(opening));
    toggle.textContent = opening ? "Close chat" : label;
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && panel.classList.contains("open")) toggle.click();
  });

  document.body.appendChild(panel);
  document.body.appendChild(toggle);
})();
