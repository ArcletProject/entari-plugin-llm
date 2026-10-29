---
name: satori-output
description: Use when the final answer needs richful element (e.g. image, at, face, etc.)
---

# Satori Output

When this skill is active, the final answer allow to use Satori Element style (similar to HTML document).


## Rules

1. Return only the satori-style. Do not add prose before or after it.
2. Do not wrap the document in Markdown code fences.
3. Do not copy or mention the `<skills_system>` block or any skill metadata in the answer.
4. Do not copy or mention the `<tool_call>` block or any tool call metadata in the answer.

## Example

### Text

usually, text are just text:

```html
123, 456
```

### Line Break (br)

The <br> element represents an independent line break. In Satori, do not use `\n`, use `<br/>` instead:

```html
123<br/>
456
```

### Mention User (at)

The <at> element is used to mention a user or users. You can use this if you known the user-id who want to mention:

```html
<at id="【user-id】" /> Hello!
```

| ATTRIBUTE | TYPE    | SCOPE | DESCRIPTION                                                                                        |
|-----------|---------|-------|----------------------------------------------------------------------------------------------------|
| `id`      | string? | S/R   | target user id                                                                                     |
| `name`    | string? | S/R   | target user name                                                                                   |
| `role`    | string? | S/R   | target role                                                                                        |
| `type`    | string? | S/R   | special operations, such as `all` for mentioning all members, `here` for mentioning online members |

### Resource Elements

Resource message elements represent resource files in the text. Different platforms have significant differences in their support for resource files. When sending, only `src` needs to be provided. If a platform does not support a specific resource type, the adapter should replace it with `src`. If a platform does not support sending resource message elements and other message elements simultaneously, the adapter should send them in multiple messages.

| ATTRIBUTE | TYPE    | SCOPE | DESCRIPTION               |
|-----------|---------|-------|---------------------------|
| `src`     | string  | S/R   | url of the resource       |
| `title`   | string? | S/R   | name of the resource file |

Especially, `src` accept three format:
- url: `http://example.com/1.png`
- file_uri: `file:///path/to/image`
- base64: `data:【mimetype】;base64,【b64 data】`


```html
I send a img to you: <img src="http://example.com/1.png" />
```

#### Image

The `<img>` element is used to represent an image.

In addition to the common attributes above, the following attributes are supported:

| ATTRIBUTE | TYPE    | SCOPE | DESCRIPTION                                |
|-----------|---------|-------|--------------------------------------------|
| `subType` | string? | S/R   | image subtype, such as `normal`, `sticker` |


#### Audio

The `<audio>` element is used to represent audio.

In addition to the common attributes above, the following attributes are supported:

| ATTRIBUTE  | TYPE    | SCOPE | DESCRIPTION               |
|------------|---------|-------|---------------------------|
| `duration` | number? | S/R   | audio length (in seconds) |
| `poster`   | string? | S/R   | url of the audio cover    |

#### Video

The `<video>` element is used to represent a video.

In addition to the common attributes above, the following attributes are supported:

| ATTRIBUTE  | TYPE    | SCOPE | DESCRIPTION               |
|------------|---------|-------|---------------------------|
| `duration` | number? | S/R   | video length (in seconds) |
| `poster`   | string? | S/R   | url of the video cover    |

#### File

The `<file>` element is used to represent a file.

In addition to the common attributes above, the following attributes are supported:

| ATTRIBUTE | TYPE    | SCOPE | DESCRIPTION          |
|-----------|---------|-------|----------------------|
| `poster`  | string? | S/R   | url of the thumbnail |

### Link

The `<a>` element is used to display a link. If the platform does not support links, it is recommended to display it in the form of `content (href)`.

```html
I shared this:
<a href="url">【show text】</a>
```

### Emoji

Use to show a emoji face, accept `id` (required) and `name`

| ATTRIBUTE | TYPE    | SCOPE | DESCRIPTION |
|-----------|---------|-------|-------------|
| `id`      | string? | S/R   | emoji id    |
| `name`    | string? | S/R   | emoji name  |

```html
<emoji id="123" />
```

### Styles

Styles are a lot of elements, include `b`/`strong`, `i`/`em`, `u`/`ins`, `s`/`del`, `spl`, `code`, `sup` and `sub`

### Paragraph

Paragraph will sure there is a newline between it and next element

```html
Hello,
<p>
    World!
</p>
```

### Message

| ATTRIBUTE | TYPE     | SCOPE | DESCRIPTION                       |
|-----------|----------|-------|-----------------------------------|
| `id`      | string?  | S     | message id                        |
| `forward` | boolean? | S     | whether it is a forwarded message |

The basic usage of the `<message>` element is to represent a message. Its child elements correspond to the message's content. If it has no child elements, the message will not be sent.

When a `<message>` element appears, the preceding elements are immediately treated as a message and sent. Therefore, the following two approaches are equivalent:

```html
<!-- First approach: Send two messages -->
<message>hello</message>
<message>world</message>

<!-- Second approach: Use an empty message to separate two pieces of text, but it still sends two messages -->
hello<message/>world
```

**Usually, `<message>` is unnecessary because you can send plain-text directly, except for the following cases:**

Some platforms allow you to simulate messages sent by other users:

```html
<message>
  <author id="123123123" name="Alice" avatar="url"/>
  hello world
</message>
```

On platforms that support forwarding, you can use forward with the id attribute to forward a message:

```html
<message id="123456789" forward/>
```

On platforms that support combined forwarding, you can use the `<message>` element with the forward attribute to nest other `<message>` elements for combined forwarding:

```html
<message forward>
  <message id="123456789"/>
  <message id="987654321"/>
  <!-- You can also nest simulated messages from other users in combined forwarding -->
  <message>
    <author id="123123123" name="Alice" avatar="url"/>
    hello world
  </message>
</message>
```

### Quote (quote)

The `<quote>` element is used to represent a message quote. Its child elements will be rendered as the quoted content. In theory, all features of the `<message>` element can also be used in the `<quote>` element, including child elements (to construct quoted messages) and the forward attribute (to quote combined forwarding). However, currently, no platform seems to support this.

### Author (author)

| ATTRIBUTE | TYPE    | SCOPE | DESCRIPTION |
|-----------|---------|-------|-------------|
| `id`      | string? | S     | user id     |
| `name`    | string? | S     | user name   |
| `avatar`  | string? | S     | avatar url  |

The `<author>` element is used to represent the author of a message. Its child elements will be rendered as the author's name.


### Markdown (markdown)

The `<markdown>` element is used to represent a markdown message. Its child elements will be rendered as the markdown content. 

```html
<markdown>
    # This is a markdown message
    - Item 1
    - Item 2 
</markdown>
```

### Platform-native Message Elements

Platforms may provide native message elements, but they must be prefixed with the adapter name. For example:

```html
<milky:face id="123" />
```

### Extension Attributes on Standard Elements

Platform-native attributes for standard elements can also be declared by prefixing them with the adapter name. For example:

```html
<!-- src is a standard attribute of the image element. -->
<!-- sub-type is not standardized, so it needs a prefix. -->
<img src="url1" milky:sub-type="sticker"/>
```