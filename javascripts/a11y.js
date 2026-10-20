// Fills accessibility gaps in the Material theme's markup.
(() => {
  // Tables wider than the screen scroll sideways, so keyboard users need to be
  // able to focus them (WCAG 2.1.1).
  // Name each table after its section, e.g. "Summaries table", so that
  // landmarks on the same page are distinguishable.
  const tableName = (wrap) => {
    const caption = wrap.querySelector("caption");
    if (caption) return caption.textContent.trim();
    const headings = [...document.querySelectorAll(".md-content h2, .md-content h3")];
    const before = headings.filter(
      (h) => h.compareDocumentPosition(wrap) & Node.DOCUMENT_POSITION_FOLLOWING
    );
    const heading = before[before.length - 1];
    const name = heading ? heading.textContent.replace("¶", "").trim() : "";
    return name ? `${name} table` : "Table";
  };

  const labelScrollers = () => {
    document.querySelectorAll(".md-typeset__scrollwrap").forEach((wrap) => {
      if (wrap.scrollWidth > wrap.clientWidth) {
        wrap.setAttribute("tabindex", "0");
        wrap.setAttribute("role", "region");
        wrap.setAttribute("aria-label", tableName(wrap));
      } else if (wrap.getAttribute("role") === "region") {
        wrap.removeAttribute("tabindex");
        wrap.removeAttribute("role");
        wrap.removeAttribute("aria-label");
      }
    });
  };

  const init = () => {
    // The search dialog has no accessible name.
    document.querySelectorAll('.md-search[role="dialog"]').forEach((el) => {
      el.setAttribute("aria-label", "Search");
    });
    // Each code block's copy button sits in an unnamed <nav>, which adds a
    // landmark per code block. The button itself stays labelled.
    document.querySelectorAll("nav.md-code__nav").forEach((el) => {
      el.setAttribute("role", "none");
    });
    labelScrollers();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(labelScrollers, 200);
  });
})();
